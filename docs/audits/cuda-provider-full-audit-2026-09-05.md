# Audit complet — Provider CUDA

**Repository principal :** `astorise/Magnetar`  
**Provider audité :** `astorise/Magnetar-provider-CUDA`  
**Date :** 2026-09-05  
**HEAD Magnetar audité :** `52363de99403e1ae8e437b41049983d5bd5ace05`  
**HEAD / gitlink CUDA audité :** `7ac13b19321c414864f0c17d0162ee509b51e1ea`  
**Quality CI Magnetar :** run `33982089323`, success  
**Dernier GPU Runner Smoke Test :** run #9 / `33961785474`, success sur `05624ea2d4febbf560fc27455af51cf763b68a0f` — antérieur à l'implémentation CUDA réelle

## 1. Verdict exécutif

> **CHANGES REQUIRED — le Provider CUDA est désormais une implémentation réelle, mais le baseline n'est pas encore conforme à ses propres contrats de résidence/mouvement mémoire et ne doit pas encore être considéré comme terminé ni archivé.**

Le progrès depuis l'audit intermédiaire est substantiel :

- le crate CUDA n'est plus un template ;
- `cudarc` est intégré ;
- un vrai `CudaProvider` et un vrai `CudaDevice` existent ;
- les kernels required-now sont implémentés et lancés via CUDA/NVRTC ;
- le contrat d'erreur `TensorValue` du Core a été corrigé ;
- le Provider est externalisé proprement en submodule ;
- le comportement sans driver CUDA a été durci après détection d'un crash par la CI ;
- le Change OpenSpec `implement-cuda-provider-baseline` existe et structure le chantier.

Mais l'audit complet trouve **deux P0 directs dans l'implémentation CUDA** et un **blocker d'intégration first-native** :

1. les kernels effectuent implicitement les copies H2D/D2H alors que la spec CUDA exige que les changements de résidence soient explicites ;
2. le Memory Manager enregistre des allocations `Device` alors que le stockage persistant du `TensorResourceId` est encore un `HostTensor` ; les allocations CUDA réelles sont seulement temporaires et hors de ce ledger ;
3. le datapath first-native reste encore Host/Reference-CPU-shaped à ses frontières de dispatch, ce qui empêche un vrai pipeline CUDA `Opaque -> Opaque` sans round-trip Host.

Tant que les deux premiers points ne sont pas corrigés, **le Change `implement-cuda-provider-baseline` ne doit pas être archivé comme baseline conforme**. Le troisième doit être fermé avant de présenter CUDA comme un backend first-native réellement utilisable de bout en bout.

## 2. Périmètre et méthode

L'audit a relu l'état Git courant des deux repositories et revalidé :

- le gitlink `providers/cuda` ;
- le `Cargo.toml` et la dépendance `cudarc` ;
- `provider.rs`, `device.rs`, `executor.rs`, `kernels.rs`, `advertisements.rs`, `error.rs` ;
- les tests CUDA/conformance ;
- le contrat `ProviderExecutionApi` côté Core ;
- le chemin first-native dans `magnetar-runtime/src/first_native_runtime.rs` ;
- le Change OpenSpec `implement-cuda-provider-baseline` ;
- les exigences `device-resident-resource` / Provider roadmap ;
- la CI standard et l'historique du runner GPU.

Le verdict est basé sur le code publié aux SHAs ci-dessus, pas sur des intentions, branches locales ou résultats manuels non reproductibles depuis GitHub Actions.

## 3. Revalidation de l'audit intermédiaire

| Point précédent | État | Conclusion |
|---|---|---|
| Canal d'erreur `TensorValue` incomplet | ✅ Fermé côté Core | `write_tensor_value` est désormais fallible et le chemin admitted distingue admission mémoire et erreur Provider |
| Provider hardcodé Reference CPU lors de la matérialisation | 🟡 Partiellement fermé | la résolution Provider/ModelInstance a été généralisée, mais le dispatch first-native reste Host/Reference-CPU-shaped |
| Pas de Change OpenSpec CUDA | ✅ Fermé | `implement-cuda-provider-baseline` existe et définit le baseline |
| Aucun vrai Provider CUDA | ✅ Fermé | Provider, Device, executor et kernels CUDA sont présents |
| `HostTensor` dans `reference_cpu` | ❌ Ouvert | le contrat générique `TensorValue::Host` dépend toujours de `reference_cpu::HostTensor` |
| Pas de profil/conformance CUDA réelle | 🟡 Partiel | tests et profil existent, mais la validation hardware actuelle n'est pas encore un gate CI reproductible sur le code courant |
| GPU runner uniquement infrastructure | ❌ Toujours vrai pour le code courant | le dernier run GPU vert précède l'implémentation CUDA réelle |

## 4. État de l'implémentation CUDA

Le Provider courant est une vraie implémentation :

- `CudaProvider` découvre un Device et expose metadata/health/advertisements ;
- `CudaExecutor` implémente `ProviderExecutionApi` ;
- `cudarc` fournit driver/NVRTC/launch ;
- les kernels CUDA sont effectivement compilés/lancés ;
- les advertisements restent bornés à F32 + contiguous ;
- aucun `QwenCudaProvider` ou autre Provider lié à une famille de modèle n'a été introduit ;
- les handles/pointeurs CUDA restent privés au Provider.

Cette direction architecturale est saine.

Le problème principal n'est donc plus « CUDA n'existe pas » mais **la fidélité entre ce que Runtime croit de la résidence et ce que le Provider fait réellement**.

---

## 5. P0 — mouvements H2D/D2H implicites à l'intérieur des kernels

### Constat

`src/kernels.rs` utilise directement, dans les chemins de calcul :

- `clone_htod(...)` pour envoyer les entrées Host vers GPU ;
- une allocation Device temporaire pour le résultat ;
- le lancement CUDA ;
- `clone_dtoh(...)` pour récupérer immédiatement le résultat en Host ;
- retour d'un `HostTensor`.

Le pattern réel est donc :

```text
HostTensor
  -> H2D caché dans le kernel
  -> allocation CUDA temporaire
  -> kernel
  -> D2H caché dans le kernel
  -> HostTensor
```

### Contrat violé

Le Change `implement-cuda-provider-baseline` contient une exigence **Explicit Data Movement** :

- un input Host ne doit pas être silencieusement uploadé en mémoire Device ;
- un output Device ne doit pas être silencieusement téléchargé en Host ;
- un changement de domaine de résidence doit passer par une opération de data movement explicite.

Les kernels actuels font précisément le contraire.

### Impact

- chaque node GPU effectue un round-trip Host inutile ;
- le graphe ne peut pas distinguer calcul et mouvement ;
- les politiques de fallback/staging ne peuvent pas gouverner le transfert ;
- l'observabilité et la causalité ne peuvent pas auditer les changements de résidence ;
- une future interdiction de host staging serait contournée par le Provider ;
- les métriques de performance/mémoire deviennent trompeuses.

### Correctif requis

Le Provider doit conserver les ressources Device entre les kernels et résoudre les buffers via `TensorResourceId`. Les H2D/D2H doivent devenir des opérations de data movement explicites, planifiées/autorisées par Runtime.

**Sévérité : P0.**

---

## 6. P0 — le Memory Manager déclare `Device` mais la ressource persistante est Host

### Constat

`CudaExecutor` possède encore un stockage du type conceptuel :

```text
BTreeMap<TensorResourceId, HostTensor>
```

`write_tensor(...)` écrit dans ce stockage Host et `read_tensor(...)` relit le `HostTensor`.

Mais le chemin `write_tensor_admitted(...)` demande au Runtime Memory Manager une admission avec une **placement Device**, puis stocke malgré tout le tensor dans la map Host.

Les vraies allocations CUDA sont créées ensuite dans `kernels.rs`, au moment de chaque invocation, puis détruites après D2H. Elles ne sont pas l'allocation persistante attachée au `TensorResourceId` et ne sont pas celles suivies par le ledger Runtime.

### Incohérence d'autorité

Runtime croit :

```text
TensorResourceId X -> Device allocation suivie par Memory Manager
```

La réalité est :

```text
TensorResourceId X -> HostTensor persistant
                  -> allocations CUDA transitoires non représentées dans le ledger
```

### Contrats violés

Cela contredit directement :

- `device-resident-resource` ;
- l'exigence CUDA Device Memory Accounting ;
- le principe selon lequel Runtime Memory Manager est l'autorité d'admission/accounting ;
- la promesse que Resource Residency reflète le vrai domaine mémoire sans exposer le pointeur natif.

### Correctif requis

Le Provider doit maintenir une table privée :

```text
TensorResourceId -> CudaAllocation { DevicePtr privé, taille, device, état }
```

Runtime ne voit jamais le pointeur, mais son admission/résidence doit correspondre à cette allocation réelle. La libération d'un `TensorResourceId` doit libérer le buffer Device correspondant et mettre à jour le ledger de façon cohérente.

**Sévérité : P0.**

---

## 7. P0 d'intégration — le first-native datapath n'est toujours pas device-resident

Le Core a été amélioré depuis l'audit intermédiaire : la résolution de certains chemins de matérialisation n'est plus systématiquement épinglée à Reference CPU.

Cependant le chemin de dispatch first-native conserve encore des hypothèses CPU/Host importantes :

- `execute_qwen_graph_nodes` lit via `read_tensor_value`, puis appelle `into_host(...)` pour les entrées de node ;
- les bindings de dispatch sont donc matérialisés en `HostTensor` avant calcul ;
- le helper de dispatch reste structuré autour de `dispatch_reference_cpu_operator_multi` ;
- les Resource Affinities / Kernel resources de ce chemin restent encore construits avec des hypothèses Reference CPU / `KernelMemoryClass::Host` à des points significatifs ;
- les outputs sont lus comme valeurs Host.

Un vrai pipeline CUDA devrait permettre :

```text
MatMul GPU -> TensorValue::Opaque / Resource Device
           -> RMSNorm GPU sur le même Device
```

sans `into_host()` entre les deux.

Aujourd'hui, un Provider qui implémenterait correctement la résidence Device se heurterait encore à ces frontières Host.

### Classement

- **P0 avant de présenter CUDA comme first-native end-to-end réellement utilisable.**
- Le Change CUDA actuel peut séparer ce travail de son incrément Provider, mais cela ne doit pas conduire à appeler l'ensemble Magnetar+CUDA « device-resident » avant fermeture.

---

## 8. P1 — `TensorValue::Opaque` est accepté comme succès sans ressource

Dans le Provider actuel :

- `read_tensor_value` retourne des valeurs Host ;
- pour `write_tensor_value(..., TensorValue::Opaque)`, le Provider ne possède pas de buffer associé et peut retourner succès sans matérialiser quoi que ce soit ;
- même problème sur le chemin admitted.

Le Provider ne produit pas encore lui-même `Opaque`, ce qui limite l'impact actuel. Mais dès que le datapath devient réellement device-resident, ce comportement devient dangereux : **un appel peut être déclaré réussi alors qu'aucune ressource n'a été créée ou conservée**.

### Correctif

Avant d'accepter `Opaque` :

- soit il correspond à une ressource Provider-private déjà existante ;
- soit l'opération échoue explicitement/fail-closed avec `ProviderExecutionError`.

Jamais de `Ok(())` silencieux sans état.

**Sévérité : P1 aujourd'hui ; deviendrait P0 dès activation de la voie Opaque réelle.**

---

## 9. P1 — Provider `Available` et kernels annoncés sans executor exécutable

`CudaProvider::new()` protège désormais les environnements sans driver/NVRTC, ce qui est positif. Mais la dégradation peut laisser :

- un Device découvert ;
- `health() == Available` ;
- des kernel advertisements présents ;
- `execution_api() == None` si l'initialisation de l'executor/NVRTC échoue.

Cette combinaison n'est pas une readiness cohérente : Runtime peut voir des kernels sélectionnables alors que l'API nécessaire à leur exécution n'existe pas.

### Correctif recommandé

Invariant à imposer :

```text
Provider Available + kernel advertised => execution_api() is Some
```

Sinon :

- Provider `Degraded`/`Unavailable`, ou
- zéro advertisement exécutable.

Éviter également de transformer un panic interne inattendu en simple « executor absent » sans diagnostic fort : `catch_unwind` est approprié pour le panic connu de chargement cudarc, mais il ne doit pas masquer des bugs arbitraires du Provider.

**Sévérité : P1.**

---

## 10. P1 — aucune CI GPU ne valide encore le code CUDA courant

La CI `Quality` du HEAD Magnetar `52363de...` est entièrement verte : run `33982089323`.

Mais ce n'est pas une preuve d'exécution CUDA : sur `ubuntu-latest`, les tests hardware se replient sur le chemin « pas de GPU »/skip.

Le dernier `GPU Runner Smoke Test` vert est :

- run #9 ;
- `33961785474` ;
- HEAD `05624ea2...` ;
- état où `providers/cuda` était encore un template.

Il n'existe donc pas encore de run GitHub Actions GPU vert sur :

- Magnetar `52363de...` ;
- Provider CUDA `7ac13b1...`.

Le fichier `gpu-runner-smoke.yml` porte même encore des commentaires/labels disant que CUDA est un « empty template » et que le job ne valide aucun comportement CUDA réel.

### Correctif requis

Déclencher puis rendre reproductible un gate GPU qui exécute au minimum :

- Device discovery ;
- allocation/free ;
- H2D/D2H explicite ;
- kernels réels ;
- conformance numérique ;
- structured errors ;
- tests residency/ledger.

**Sévérité : P1 readiness blocker.**

---

## 11. P1 — le profil CUDA peut encore passer sans valider le hardware

Les tests Provider invoquent les profils génériques et `ProviderConformanceProfile::Cuda`, ce qui est une bonne fondation.

Cependant le framework Core considère encore les profils hardware comme opt-in/skippables selon le contexte. Un test qui vérifie seulement que le report global est « conformant » peut donc être vert alors que le profil CUDA hardware a été `Skipped`.

En parallèle, `tests_conformance.rs` contient de vrais tests numériques GPU-gated, mais ils n'ont pas encore été exécutés dans GitHub Actions sur la version courante.

### Correctif requis

Le gate GPU doit exiger explicitement :

```text
provider-hardware-cuda == Passed
```

et non simplement « le report ne contient pas de Failed ».

**Sévérité : P1.**

---

## 12. P1 — la catégorie OOM stable est perdue au boundary Kernel

`CudaProviderError` définit correctement une variante :

```text
OutOfDeviceMemory { requested, available }
```

avec un code stable `cuda-out-of-device-memory`.

Mais la conversion vers `KernelError` regroupe l'OOM dans un `KernelExecutionFailed(...)` générique.

La spec CUDA exige qu'une allocation impossible / OOM soit représentée par une catégorie stable et structurée. La catégorie interne existe donc, mais elle est perdue avant de remonter au consommateur Kernel/Runtime.

### Correctif

Conserver le code/catégorie OOM jusqu'au boundary Runtime, ou ajouter une variante structurée Kernel/Provider appropriée.

Ajouter un test d'échec réel/simulé vérifiant le code observé par l'appelant, pas uniquement la variante interne CUDA.

**Sévérité : P1.**

---

## 13. P2 — découverte limitée à GPU ordinal 0 vs wording de la spec

`discover_primary_cuda_device()` ouvre actuellement le Device ordinal `0` et le Provider stocke un seul Device/context/executor.

Le design autorise explicitement un baseline single-GPU, ce qui est raisonnable. Mais la spec emploie un wording plus fort indiquant que chaque Device CUDA compatible discoverable doit être exposé.

### Décision à prendre

Soit :

- modifier la spec du baseline pour dire explicitement « primary/single device » ;
- soit énumérer tous les Devices tout en gardant le multi-device placement hors scope.

**Sévérité : P2 tant que le single-GPU est assumé comme scope.**

---

## 14. P2 — `TensorValue::Host` dépend encore de `reference_cpu::HostTensor`

Le contrat générique Core reste conceptuellement :

```rust
TensorValue::Host(crate::reference_cpu::HostTensor)
```

CUDA dépend désormais réellement de ce contrat. Ce qui était auparavant une dette théorique devient donc une dépendance inter-Provider visible.

### Recommandation

Déplacer le type Host générique vers `tensor`/`host_tensor`/un module neutre puis re-exporter depuis `reference_cpu` si une compatibilité est nécessaire.

**Sévérité : P2/P1 API design ; à traiter avant multiplication des Providers.**

---

## 15. P2 — supply-chain du submodule CUDA hors `cargo-deny` racine

Le crate CUDA ajoute une nouvelle dépendance critique : `cudarc`.

Comme les submodules ont leur propre workspace/lockfile, le `cargo deny` du repository principal ne traverse pas automatiquement `providers/cuda/Cargo.lock`.

Le Change OpenSpec le reconnaît déjà dans ses tâches.

### Recommandation

Ajouter, dans le gate provider/submodule :

```text
cargo deny --manifest-path providers/cuda/Cargo.toml ...
```

ou un workflow équivalent exécuté dans le submodule.

**Sévérité : P2 supply-chain.**

---

## 16. P2 — workflow GPU devenu documentairement faux

`.github/workflows/gpu-runner-smoke.yml` affirme encore :

- `providers/cuda is currently an empty template` ;
- `no CUDA-specific behavior yet worth gating` ;
- étape `Build and test the CUDA Provider template`.

Ces assertions sont désormais fausses.

Le job lui-même est utile et devrait maintenant devenir le gate de conformance hardware plutôt qu'un simple smoke de runner.

**Sévérité : P2 documentation/CI intent.**

---

## 17. Points solides à conserver

L'audit valide positivement plusieurs décisions importantes :

### Provider externalisé

Le gitlink `providers/cuda` pointe sur `7ac13b1...`. La dépendance est dans le bon sens :

```text
magnetar-provider-cuda -> magnetar-runtime
```

Le Core ne dépend pas du repository CUDA.

### CUDA réel, pas de CPU fallback masqué

Les kernels sont de vrais launches CUDA/NVRTC. Le Provider n'est pas un wrapper qui appelle Reference CPU lorsque CUDA est disponible.

### Scope numérique borné

F32 + contiguous seulement est une très bonne limite pour le premier incrément. Il vaut mieux rendre cette petite surface correcte et device-resident avant d'ajouter BF16/F16/quantization/FlashAttention.

### Abstraction model-family respectée

Le Provider reste `CudaProvider` et annonce des Operators/Kernels génériques ; aucune connaissance Qwen/Llama n'est introduite dans le Provider.

### Handles CUDA privés

Les contextes, streams, modules et device pointers ne traversent pas les contrats Runtime publics.

### Graceful absence de CUDA améliorée

La CI sans driver a trouvé un vrai défaut : cudarc panicait lorsque la librairie driver/NVRTC était absente. Le HEAD `7ac13b1` protège ce cas et le main Quality est revenu au vert. C'est un bon exemple de CI utile.

### Tests déjà présents

Le Provider contient des tests numériques et des tests de conformance. Le problème est désormais surtout de les exécuter dans un environnement GPU reproductible et de durcir leurs assertions, pas de repartir de zéro.

---

## 18. État de la CI

### Quality standard

Pour Magnetar `52363de99403e1ae8e437b41049983d5bd5ace05` :

- run `33982089323` ;
- statut `completed` ;
- conclusion `success`.

Ce résultat valide : build, tests sans GPU, intégration submodule, OpenSpec, lint/format et autres gates standards.

### GPU

Le dernier run GPU vert connu est #9 sur `05624ea2...`, avant les commits réels CUDA.

**Conclusion CI :**

```text
CPU/static integration: GREEN
current real CUDA hardware execution: NOT YET PROVEN IN GITHUB ACTIONS
```

---

## 19. État du Change OpenSpec

`implement-cuda-provider-baseline` est encore actif, ce qui est correct.

Il ne doit pas être archivé maintenant.

Au minimum, les tâches de validation hardware/runner et OOM ne sont pas toutes closes, et l'audit ajoute surtout les écarts normatifs de résidence/mouvement ci-dessus.

La spec canonique `openspec/specs/cuda-provider/spec.md` ne doit donc pas être promue comme baseline réalisée avant fermeture des P0.

---

## 20. Ordre de correction recommandé

1. **P0 — rendre le stockage CUDA réellement device-resident** : `TensorResourceId -> allocation CUDA privée`.
2. **P0 — aligner Memory Manager avec l'allocation physique réelle** : aucune `MemoryPlacement::Device` pour une ressource seulement stockée en Host.
3. **P0 — extraire H2D/D2H hors des kernels** et les représenter comme data movement explicite.
4. **P1 — faire échouer `Opaque` si aucune vraie ressource Provider-private n'existe**, puis supporter réellement `Opaque` avec la nouvelle table Device.
5. **P1 — rendre health/advertisements/execution_api atomiquement cohérents**.
6. **P1 — exécuter le vrai gate GPU sur le HEAD courant**, pas sur le template historique.
7. **P1 — exiger explicitement `provider-hardware-cuda == Passed`**.
8. **P1 — préserver la catégorie OOM jusqu'au caller Runtime/Kernel**.
9. **P0 intégration — généraliser le first-native dispatch** pour chaîner deux kernels CUDA sans `into_host` ni affinité Reference CPU imposée.
10. **P2 — nettoyer discovery wording, HostTensor namespace, cargo-deny submodule et workflow GPU obsolète.**

---

## 21. Critères de sortie du prochain audit

Le prochain audit peut passer à `APPROVE` pour le baseline CUDA lorsque les preuves suivantes existent :

- un `TensorResourceId` Device correspond à une vraie allocation CUDA persistante ;
- le Memory Manager suit cette allocation et sa libération sans divergence ;
- `read_tensor_value` peut représenter correctement une ressource non-host-visible ;
- deux kernels CUDA consécutifs consomment la même ressource sans D2H/H2D intermédiaire ;
- les H2D/D2H nécessaires sont explicites, auditables et policy-controlled ;
- aucun `Ok(())` ne perd un `TensorValue::Opaque` ;
- Provider health/advertisements ne déclarent pas exécutable un executor absent ;
- OOM remonte sous une catégorie structurée stable ;
- `ProviderConformanceProfile::Cuda` est réellement `Passed` sur GPU ;
- un run `arc-gpu-magnetar` vert existe sur le commit CUDA audité ;
- le first-native datapath peut utiliser un binding CUDA sans repasser par `HostTensor` entre nodes du même Device ;
- le Change OpenSpec ne contient plus de tâche hardware critique ouverte.

---

## 22. Décision finale

> **NE PAS ARCHIVER `implement-cuda-provider-baseline` ET NE PAS PRÉSENTER LE PROVIDER COMME BASELINE DEVICE-RESIDENT CONFORME À CE STADE.**

Le chantier est néanmoins sur une bonne trajectoire : les abstractions générales sont suffisamment proches de la cible, les kernels CUDA sont réels et le Provider est correctement externalisé. Les défauts critiques sont maintenant très localisés : **residency/accounting/data movement**, puis **intégration first-native**.

La priorité n'est pas d'ajouter davantage de kernels ou d'optimisations. La prochaine preuve d'architecture doit être :

```text
Host input
  -> data movement H2D explicite
  -> TensorResourceId réellement Device-resident
  -> CUDA kernel A
  -> même ressource Device consommée directement par CUDA kernel B
  -> D2H explicite uniquement si l'API finale le requiert
```

avec le Memory Manager, les erreurs et la conformance décrivant exactement ce qui s'est réellement passé.
