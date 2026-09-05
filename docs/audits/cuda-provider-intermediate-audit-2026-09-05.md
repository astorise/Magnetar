# Audit intermédiaire — Provider CUDA

**Repository principal :** `astorise/Magnetar`  
**Provider :** `astorise/Magnetar-provider-CUDA`  
**Date :** 2026-09-05  
**HEAD Magnetar audité :** `893c0a8644d8b481aedb38328636de74f85d3f68`  
**Gitlink `providers/cuda` audité :** `e11935e29ffbc961bd1aa1c9d076ff165c2c4d53`  
**GPU smoke validé :** workflow `GPU Runner Smoke Test` #9, run `33961785474`, success

## 1. Verdict exécutif

> **GO CONDITIONNEL — l'infrastructure permettant de commencer le Provider CUDA est saine, mais deux contrats Core doivent être corrigés avant d'investir fortement dans l'implémentation des kernels et de la mémoire device-resident.**

Le code publié de `Magnetar-provider-CUDA` est encore un template `cargo new` : il ne contient actuellement ni binding CUDA, ni Device discovery, ni allocation GPU, ni Kernel, ni implémentation `ProviderExecutionApi`. L'audit est donc volontairement un **audit de readiness d'architecture et d'intégration**, pas encore un audit numérique des kernels CUDA.

Le point positif est important : l'équipe a correctement préparé le terrain matériel. Le runner ARC `arc-gpu-magnetar` est accessible, le job s'exécute dans `nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04`, `nvidia-smi`, `nvcc` et le toolkit sont vérifiés, le submodule est checkouté et le crate compile/teste sur le runner GPU. Le smoke #9 est vert après correction des problèmes `git` et `safe.directory`.

En revanche, deux blockers architecturaux apparaissent avant le premier vrai datapath CUDA :

1. **P0 — le first-native runtime reste CPU-centric et force encore des matérialisations HostTensor.** Un tensor CUDA correctement représenté par `TensorValue::Opaque` ne peut pas traverser le chemin Qwen actuel de bout en bout sans host round-trip / erreur de résidence. Plusieurs zones de production épinglent également `REFERENCE_CPU_PROVIDER_NAME` et `REFERENCE_CPU_DEVICE_ID`.
2. **P0 — la voie `TensorValue` n'a pas encore un canal d'erreur Provider cohérent avec le besoin CUDA.** `ProviderExecutionApi::write_tensor` a bien été corrigé pour retourner `Result<(), ProviderExecutionError>`, mais `write_tensor_value` retourne encore `()` et son défaut est un no-op ; `write_tensor_value_admitted` retourne `MemoryError`, ce qui ne permet pas de représenter proprement un échec CUDA de copie/allocation/exécution comme erreur Provider.

Ces deux points doivent être traités **avant** de faire de la mémoire GPU et des kernels un investissement important, sinon l'implémentation externe risque d'être contrainte par un contrat host-only puis réécrite.

## 2. État réellement publié du Provider CUDA

Le dépôt `astorise/Magnetar-provider-CUDA` contient actuellement :

- `.gitignore` ;
- `Cargo.toml` avec aucun dependency ;
- `Cargo.lock` ;
- un README déclarant explicitement le statut `Empty template` ;
- `src/lib.rs` avec uniquement le `add(2, 2)` généré par `cargo new`.

Aucune branche de développement ou PR CUDA n'est publiée dans ce dépôt au moment de l'audit. Le submodule de `Magnetar/main` pointe toujours sur `e11935e29ffbc961bd1aa1c9d076ff165c2c4d53`.

**Conséquence :** il n'est pas encore possible d'émettre un avis sur la qualité d'un contexte CUDA, d'un allocator, de kernels, de streams, de synchronisation ou de mapping d'erreurs : ces éléments n'existent pas encore dans l'état Git observable.

## 3. Infrastructure GPU — état positif

Le workflow `.github/workflows/gpu-runner-smoke.yml` est correctement séparé de la CI CPU habituelle et vérifie réellement l'environnement GPU :

- runner self-hosted `arc-gpu-magnetar` ;
- job container CUDA ;
- `--gpus all` ;
- `nvidia-smi` ;
- présence de `nvcc` ;
- version `nvcc --version` ;
- présence de `${CUDA_HOME:-/usr/local/cuda}` ;
- Rust toolchain du repository ;
- checkout du gitlink `providers/cuda` ;
- `cargo test --locked --manifest-path providers/cuda/Cargo.toml`.

Le run #9 est intégralement vert, y compris l'étape `Verify GPU is visible`.

### Limite actuelle du smoke

Ce run prouve **le runner et le toolkit**, pas encore le Provider :

- le crate n'a aucun binding CUDA ;
- aucun appel driver/runtime CUDA n'est compilé ;
- aucun device n'est ouvert ;
- aucune allocation GPU n'est créée ;
- aucun kernel n'est lancé ;
- aucun résultat n'est comparé à Reference CPU.

C'est normal à ce stade, mais le nom/objectif du gate devra évoluer dès le premier code CUDA.

## 4. P0 — le first-native datapath n'est pas encore consommable par un Provider CUDA device-resident

### 4.1 Hard-coding Reference CPU

`magnetar-runtime/src/first_native_runtime.rs` contient encore plusieurs dépendances de production directes à :

- `REFERENCE_CPU_PROVIDER_NAME` ;
- `REFERENCE_CPU_DEVICE_ID` ;
- des `ResourceAffinity` explicitement liées au Provider CPU ;
- `WeightMaterializationTransaction::begin()` qui résout explicitement Reference CPU ;
- `KvUpdateTransaction::begin()` qui résout explicitement Reference CPU ;
- des prepared-kernel fixtures liés au Provider/Device CPU.

Ce choix était correct pour le baseline first-native CPU. Il devient un blocker d'intégration dès que le même moteur doit exécuter un plan dont les bindings sélectionnent CUDA.

### 4.2 `TensorValue::Opaque` existe, mais le chemin Qwen le rematérialise encore en host

Le contrat `TensorValue` est une bonne base :

```text
Host(HostTensor)
Opaque
```

`Opaque` permet à un Provider d'indiquer honnêtement : « la ressource existe dans mon domaine mémoire, mais je ne fournis pas de représentation host ».

Le problème est situé plus haut : `execute_qwen_graph_nodes` lit bien via `read_tensor_value`, mais appelle encore `TensorValue::into_host` à plusieurs frontières qui se trouvent dans le chemin normal :

- poids ;
- entrée de chaque node/kernel ;
- historique KV pour concaténation ;
- matérialisation finale.

Le commentaire source reconnaît explicitement que chaque résolution d'entrée de Kernel est une frontière `into_host`.

Pour Reference CPU, cela est transparent. Pour CUDA :

```text
CUDA MatMul output -> TensorValue::Opaque -> next RMSNorm input -> into_host() -> ResidencyUnavailable
```

Ce comportement viole l'objectif `device-resident-resource` pour un pipeline GPU homogène : deux kernels compatibles sur GPU0 doivent pouvoir chaîner sans host round-trip obligatoire.

### Décision recommandée

Ne pas rendre le Provider CUDA responsable de contourner ce problème. Le correctif appartient au Core :

- le dispatch de node doit transmettre des `TensorResourceId` / descriptors au Provider ;
- le Provider doit résoudre ses pointeurs/handles privés depuis ces ids ;
- aucune `HostTensor` ne doit être exigée entre deux kernels du même Provider/Device ;
- la matérialisation host doit devenir une vraie opération de data movement explicite, uniquement lorsque le plan la demande.

**Gate : P0 avant E2E CUDA et avant optimisation de kernels.**

## 5. P0 — canal d'erreur incomplet sur le chemin `TensorValue`

Le commit `4407a585...` a correctement fermé l'ancien problème #41 pour les mutations historiques :

- `write_tensor(...) -> Result<(), ProviderExecutionError>` ;
- `release_tensor(...) -> Result<bool, ProviderExecutionError>` ;
- propagation et rollback sur les chemins de poids/KV/unload.

Mais le chemin destiné précisément aux ressources device-resident reste asymétrique :

### `write_tensor_value`

Le trait expose encore :

```rust
fn write_tensor_value(&self, id: TensorResourceId, value: TensorValue)
```

Le default implementation est un no-op.

Pour CUDA, un write peut échouer pour des raisons réelles :

- allocation GPU ;
- H2D / D2D ;
- contexte/device perdu ;
- adresse/ressource invalide ;
- erreur asynchrone remontée lors d'une synchronisation ;
- OOM ;
- stream en échec.

Un `()` ne peut pas porter ces erreurs.

### `write_tensor_value_admitted`

La méthode retourne actuellement `Result<(), MemoryError>`. Cela permet un échec du ledger mémoire, mais ne modélise pas proprement un échec natif du Provider.

### Risque

Le premier Provider qui a réellement besoin de `TensorValue::Opaque` serait obligé soit :

- d'écraser une erreur CUDA dans `MemoryError` ;
- de stocker une erreur différée ailleurs ;
- de panic ;
- ou de rendre certaines opérations artificiellement infallibles.

Les quatre options sont mauvaises.

### Décision recommandée

Unifier les mutations `TensorValue` avec la correction #41 avant l'implémentation CUDA :

- `write_tensor_value -> Result<(), ProviderExecutionError>` ;
- `write_tensor_value_admitted` doit distinguer admission Runtime et mutation Provider, soit avec un error enum composite, soit avec une API transactionnelle qui garde les catégories séparées ;
- les release/movement device-resident doivent avoir la même discipline ;
- tests injectant un échec H2D/D2D simulé et vérifiant rollback Memory Manager + Provider storage.

**Gate : P0 avant vraie gestion mémoire CUDA.**

## 6. P1 — aucun OpenSpec Change CUDA dédié n'existe encore

`openspec/changes/` ne contient aujourd'hui que l'archive et `define-provider-prepared-kernel-execution-contract`. Le README du Provider CUDA indique lui-même qu'aucune spec CUDA dédiée n'existe et que le travail n'était pas encore scheduled.

Le début du chantier change cette situation.

### Pourquoi c'est important maintenant

Les décisions CUDA qui semblent « détails d'implémentation » deviennent rapidement des contrats difficiles à changer :

- Driver API vs Runtime API ;
- un contexte par Device vs autre ownership ;
- primary context ou contexte privé ;
- synchronisation blocking initiale vs streams ;
- représentation des device allocations ;
- pinned host memory ;
- mapping stable des devices ;
- compute capability minimum ;
- dtype initial ;
- layout initial ;
- stratégie cuBLAS ;
- kernels custom minimum ;
- comportement de cancellation ;
- erreurs async ;
- politique de fallback ;
- destruction/context-loss ;
- multi-GPU explicitement hors scope ou non.

### Scope recommandé pour le premier Change

Un `implement-first-cuda-provider` minimal devrait viser :

1. **single NVIDIA Device** sélectionné par Runtime ;
2. `F32` uniquement ;
3. contiguous layout uniquement ;
4. Device discovery + metadata stable ;
5. Provider registration et advertisements ;
6. Provider-owned device allocations ;
7. H2D / D2H explicit data movement ;
8. `TensorValue::Opaque` pour resident GPU ;
9. un kernel simple puis MatMul ;
10. synchronisation déterministe / blocking pour la première preuve ;
11. structured CUDA error mapping ;
12. correctness comparison against Reference CPU ;
13. unload/release/context-loss tests ;
14. E2E avec au moins deux kernels GPU consécutifs sans host round-trip.

À laisser hors scope du premier incrément : flash attention, quantization, multi-GPU, peer access, async overlapping, autotuning, fusion avancée, runtime compilation.

**Sévérité : P1 de gouvernance/architecture ; à créer avant multiplication des fichiers CUDA.**

## 7. P1 — le profil de conformance CUDA est encore un placeholder

La roadmap est correcte sur le papier : `ProviderRoadmapPhase::Cuda` exige :

- `ProviderCore` ;
- `ProviderDataMovement` ;
- `Cuda` (`provider-hardware-cuda`).

Mais `run_provider_conformance` traite actuellement les profils hardware (`Cuda`, `Metal`, `OpenVino`, `Qnn`, `WebGpu`) comme **optional hardware profile** et les marque `Skipped` avec le message « hardware-specific profiles are opt-in and not part of default CI ».

Autrement dit : aujourd'hui `ProviderConformanceProfile::Cuda` existe comme identifiant, pas encore comme suite de conformance matérielle.

### Gate recommandé

Avant de déclarer le Provider CUDA « working » :

- implémenter un vrai profil `provider-hardware-cuda` ;
- le lancer sur `arc-gpu-magnetar` ;
- séparer clairement tests hardware-independent et hardware-required ;
- comparer les résultats numériques aux kernels Reference CPU avec tolérance explicite ;
- vérifier device residency et absence de host copy entre deux kernels GPU ;
- vérifier H2D/D2H explicite ;
- vérifier structured errors ;
- vérifier release et memory ledger.

**Sévérité : P1 ; blocker de readiness, pas blocker du premier commit.**

## 8. P1 — le mode d'intégration du Provider externe doit être décidé

L'architecture impose correctement :

```text
providers/cuda -> magnetar-runtime
magnetar-runtime -X-> providers/cuda
```

Mais il faut encore choisir comment l'application finale obtient et enregistre `CudaProvider`.

### Option A — statique / feature de l'embedder

Un binaire d'intégration dépend de :

- `magnetar-runtime` ;
- `magnetar-provider-cuda` ;

puis construit le Runtime avec `register_provider(Arc::new(CudaProvider...))`.

Cette option est simple pour le premier incrément et respecte l'absence de dépendance `runtime -> cuda`.

### Option B — dynamic Provider ABI

Le Core définit déjà le modèle ABI, mais le loader dynamique actuel retourne encore `UnsupportedDynamicAbi` après validation du path. Le chargement réel de `.so` n'est donc pas implémenté.

Si l'équipe veut que CUDA soit une vraie extension chargeable dynamiquement dès le départ, **c'est un chantier Core supplémentaire** à planifier ; le Provider CUDA seul ne peut pas le résoudre.

### Recommandation

Pour le premier E2E CUDA : **registration statique par un embedder/test harness**. Ne pas coupler le lancement des kernels CUDA à l'implémentation simultanée du dynamic ABI.

Puis traiter le dynamic loading séparément.

## 9. P1 — `TensorValue::Host` dépend encore de `reference_cpu::HostTensor`

Le type générique est actuellement :

```rust
TensorValue::Host(crate::reference_cpu::HostTensor)
TensorValue::Opaque
```

Cela fonctionne, mais fait dépendre un contrat Provider générique d'un type hébergé dans le module `reference_cpu`.

Le Provider CUDA aura tôt ou tard besoin d'une représentation host pour :

- upload initial de poids ;
- H2D/D2H ;
- validation de résultats ;
- éventuellement fallback/data movement.

S'il commence à importer `reference_cpu::HostTensor`, ce détail historique va devenir une dépendance publique de plusieurs Providers externes.

### Recommandation

Avant la première vraie dependency de `Magnetar-provider-CUDA` vers ce type :

- déplacer/renommer `HostTensor` vers un module générique (`tensor`, `host_tensor`, `portable_tensor_value`, etc.) ;
- conserver éventuellement un re-export de compatibilité depuis `reference_cpu` ;
- faire dépendre Reference CPU **et** CUDA du type host générique.

**Sévérité : P1 architecture/API ; coût très faible maintenant, coût élevé après plusieurs Providers.**

## 10. P1 — la CI principale traite encore CUDA comme template CPU-buildable

Dans `quality.yml` :

```text
Build and test the optional CUDA Provider (template today, no hardware gate needed yet)
```

Ce gate compile le crate sur `ubuntu-latest` sans GPU. C'est utile comme test de compilation host, mais ne deviendra pas suffisant dès qu'un vrai binding ou runtime path CUDA existe.

### Evolution recommandée

Conserver deux niveaux :

**CI standard, sans GPU**
- formatting ;
- clippy ;
- compilation des parties abstraites ;
- tests purs ;
- conformance metadata ;
- éventuellement feature `cuda` non activée si le linkage l'exige.

**CI GPU**
- driver/toolkit visible ;
- Device discovery ;
- allocation/free ;
- H2D/D2H ;
- kernels ;
- correctness ;
- conformance CUDA ;
- E2E device-resident.

Le workflow GPU peut rester ciblé/path-filtered plutôt que faire consommer un runner GPU à chaque changement de documentation.

## 11. P2 — version CUDA / compatibilité à formaliser

Le runner utilise actuellement CUDA **13.3.1**. C'est satisfaisant pour préparer l'environnement, mais le Provider devra distinguer :

- version de toolkit utilisée pour construire ;
- version minimale de driver supportée ;
- compute capabilities / SM supportés ;
- fonctionnalités optionnelles selon hardware ;
- comportement en absence de GPU compatible.

Le Provider ne doit pas faire de la version de l'image CI son contrat implicite de compatibilité.

## 12. Memory Manager et device residency — point de vigilance

La spec `device-resident-resource` impose déjà les bons invariants :

- résidence logique observable sans pointeur natif ;
- ressource pouvant exister sans buffer host ;
- same-device chaining sans host copy ;
- changement de domaine via data movement explicite ;
- conservation des ressources tant qu'un work in-flight les référence ;
- peer access explicite ;
- aucun native pointer dans les APIs publiques.

Le premier CUDA allocator doit donc éviter deux anti-patterns :

1. **allouer en CUDA dans le Provider sans synchroniser le ledger Runtime** ;
2. **stocker un `CUdeviceptr` / raw pointer dans `TensorResourceDescriptor` ou `TensorValue`.**

La bonne direction est :

```text
TensorResourceId -> Provider-private allocation table -> CUdeviceptr privé
```

Le Runtime ne voit que l'id, descriptor, affinity, residency et accounting.

## 13. Ordre d'implémentation recommandé

### Gate 0 — avant le Provider

1. Ouvrir le Change OpenSpec CUDA minimal.
2. Corriger le canal d'erreur `TensorValue`.
3. Décorréler `HostTensor` de `reference_cpu`.
4. Définir l'embedder/harness qui enregistrera statiquement CUDA pour le premier E2E.

### Gate 1 — Provider « alive »

5. Ajouter la dependency sur `magnetar-runtime`.
6. Choisir/documenter l'API CUDA et l'ownership du contexte.
7. Implémenter `CudaProvider` + `CudaDevice` discovery.
8. Advertise uniquement ce qui est réellement supporté.
9. Ajouter health/status/structured initialization errors.
10. GPU test : Runtime enregistre le Provider et voit le Device.

### Gate 2 — mémoire réelle

11. Provider-private allocation table par `TensorResourceId`.
12. allocate/free GPU.
13. H2D/D2H explicites.
14. `TensorValue::Opaque` pour device-resident.
15. Memory Manager ledger + residency alignés.
16. Tests OOM / invalid resource / double release / context loss.

### Gate 3 — premier compute

17. Un kernel trivial device-resident pour valider launch + lifecycle.
18. MatMul F32 contiguous.
19. Comparaison Reference CPU dans une tolérance documentée.
20. Kernel advertisement réellement conforme aux limites hardware.

### Gate 4 — datapath CUDA

21. Retirer l'hypothèse Reference CPU des zones partagées nécessaires.
22. Faire consommer les `TensorResourceId` directement par les kernels CUDA sans `into_host` inter-node.
23. Deux kernels GPU consécutifs sans host round-trip.
24. KV GPU résident sans concaténation host obligatoire.
25. logits seulement matérialisés où le plan/API le requiert.

### Gate 5 — conformance

26. Implémenter `ProviderConformanceProfile::Cuda` réel.
27. Ajouter GPU conformance au runner ARC.
28. Ajouter CUDA E2E first-model.
29. Mettre à jour `SUBMODULES.md` avec un vrai commit compatibility claim.

## 14. Ce qu'il ne faut pas faire

- Ne pas copier `ReferenceCpuExecutor` puis remplacer progressivement ses `Vec<f32>` par des buffers CUDA : cela conserverait le modèle host dans l'architecture.
- Ne pas stocker de device pointer dans les structures Runtime publiques.
- Ne pas faire de copie D2H implicite dans `read_tensor_value` pour faire « marcher » le first-native actuel : ce serait masquer le vrai problème de residency.
- Ne pas faire du CPU fallback automatique à l'intérieur du Provider CUDA.
- Ne pas introduire `QwenCudaProvider` ou des kernels qui dépendent de la famille de modèle ; le Provider implémente des Operators/Kernels portables.
- Ne pas annoncer FlashAttention/quantization/fusion dans metadata avant implémentation + conformance.
- Ne pas lancer le dynamic ABI en même temps que le premier CUDA E2E sauf besoin produit impératif.

## 15. Verdict par axe

| Axe | Verdict intermédiaire |
|---|---|
| Runner GPU / toolkit | ✅ Prêt |
| Submodule wiring | ✅ Prêt |
| Crate CUDA | ⚪ Template, pas encore auditable |
| Provider/Device public contracts | ✅ Base solide |
| Device-resident representation | ✅ `TensorValue::Opaque` existe |
| Device-resident first-native execution | ❌ P0 : host materialization / CPU pinning |
| Structured tensor mutation errors | ❌ P0 : voie `TensorValue` incomplète |
| CUDA OpenSpec scope | ⚠️ P1 : absent |
| CUDA hardware conformance | ⚠️ P1 : placeholder `Skipped` |
| External Provider integration | ⚠️ P1 : registration mode à figer |
| Generic host tensor API | ⚠️ P1 : `reference_cpu::HostTensor` fuit dans le contrat générique |
| GPU CI | ✅ smoke infrastructure ; ⚠️ pas encore gate fonctionnel |
| Dynamic loading | ⚪ Non requis pour le premier E2E ; loader réel non implémenté |

## 16. Décision finale

> **L'équipe peut continuer immédiatement sur la spec, Device discovery et le harness d'intégration. Elle ne devrait pas encore investir dans une large bibliothèque de kernels CUDA ou dans un allocator sophistiqué tant que les deux P0 Core ne sont pas fermés.**

La bonne nouvelle est que ces défauts sont découverts au meilleur moment : le Provider CUDA n'a encore aucune dette d'implémentation, le runner GPU est déjà fonctionnel, et les contrats génériques (`Provider`, `Device`, `KernelAdvertisement`, `TensorResourceId`, `TensorValue::Opaque`, Resource Affinity, Memory Manager) fournissent déjà l'essentiel du squelette.

Le prochain jalon pertinent n'est pas « MatMul CUDA rapide ». C'est :

> **un `CudaProvider` enregistré par le Runtime, un vrai `CudaDevice`, une allocation `TensorResourceId` device-resident, un petit kernel exécuté sur GPU, puis un second kernel consommant directement le résultat du premier sans D2H/H2D intermédiaire — avec erreurs structurées et Memory Manager cohérent.**

Une fois cette preuve obtenue, l'architecture CUDA sera réellement validée et l'équipe pourra accélérer sur la couverture opérateur et la performance avec beaucoup moins de risque de réécriture.
