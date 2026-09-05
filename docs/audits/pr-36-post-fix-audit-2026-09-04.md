# Audit complet post-correctifs — PR #36

**Repository:** `astorise/Magnetar`  
**Pull request:** #36 — `retour audit`  
**Date de revalidation:** 2026-09-04  
**Branche auditée:** `make-first-native-datapath-authoritative`  
**HEAD audité:** `e108a1d3910b89bd6381104dbaa857c13d86fb35`  
**Base déclarée:** `main` (`dbf4551a990efe40e4ecfaa62536dfb1bc80e29c`)

## 1. Verdict exécutif

> **APPROVE — la PR #36 est mergeable pour le périmètre borné du first-native baseline.**

Je ne retrouve plus de défaut **P0 / bloquant** dans le chemin first-native actuellement revendiqué : Qwen Component, Reference CPU Provider, F32, mono-host, KV incrémental, sampling greedy et CLI first-party.

Les deux P0 historiques autour de la matérialisation des poids et de l'autorité de readiness sont effectivement corrigés : une instance ne devient plus `Ready` avant la matérialisation complète, la matérialisation est transactionnelle, un échec rollbacke les ressources déjà staged, et les preuves de chargement/readiness sensibles ne sont plus forgeables depuis l'API publique.

La PR laisse toutefois plusieurs écarts connus et correctement tracés. Ils ne remettent pas en cause le profil borné actuel, mais ils doivent rester visibles :

- **P1 — #38** : l'exécuteur first-native générique ne propage encore qu'une sortie par nœud ; le baseline Qwen courant n'utilise que des opérateurs mono-sortie.
- **P1 — #39** : l'isolation des familles de modèles dans le Core est protégée par une allowlist CI encore non exhaustive.
- **P1 — #40** : un Component fourni par l'embedder est encore classé `DevelopmentFixture` au lieu de `ClientProvided` ; la politique de confiance ne se fonde toutefois pas sur cette métadonnée seule.
- **P1 — #41** : `ProviderExecutionApi::write_tensor` / `release_tensor` n'offrent pas encore de canal d'erreur structuré ; cela devient bloquant pour des Providers réellement faillibles, mais pas pour le Provider CPU de référence en mémoire utilisé par ce profil.
- **P2 — #42** : une `ModelInstance` abandonnée en état `Loading` ne peut pas encore être explicitement annulée/déchargée.
- **P2 — #37** : la signature cryptographique et l'authentification de l'identité éditeur restent un travail de conception ; le comportement actuel reste fail-closed via digest pinning / politique de développement explicite.

**Conclusion de merge :** aucun de ces écarts ne justifie de retenir la PR #36 dans son périmètre first-native actuel. En revanche, ils empêchent de présenter ce merge comme une validation générale de Providers matériels faillibles, d'un Core totalement family-agnostic ou d'une chaîne de supply-chain cryptographiquement signée.

## 2. Méthode de revalidation

L'audit a été repris depuis le HEAD courant, sans considérer les cases OpenSpec cochées comme une preuve suffisante. La vérification a porté sur :

1. la topologie Git et le vrai delta de la PR ;
2. l'autorité du Runtime sur l'exécution modèle ;
3. le chargement, la provenance et la matérialisation des poids ;
4. la création et la readiness des `ModelInstance` ;
5. l'autorité du Qwen WASM Component sur les graphes de production ;
6. les plans préparés, le Kernel Registry et le Provider dispatch ;
7. les Tensor Resources, le KV cache et le decode incrémental ;
8. les frontières CLI / Runtime ;
9. les formats et sous-modules externalisés ;
10. la sécurité et la supply chain ;
11. les tests négatifs et la CI ;
12. la cohérence entre les specs OpenSpec et l'implémentation effective.

La PR compte **195 fichiers modifiés** et environ **20,7 k ajouts** ; l'audit a donc été conduit par invariants architecturaux et chemins causaux, avec inspection ciblée des implémentations et des régressions qui portent réellement ces invariants.

## 3. État des retours précédents

| Sujet revalidé | État | Conclusion |
|---|---|---|
| Logits injectables par l'appelant | ✅ Fermé | Les logits sont des sorties de `execute_prefill` / `execute_decode_step`, pas des entrées caller-controlled. |
| Exécuteur modèle substituable depuis l'API publique | ✅ Fermé | L'enregistrement du `RuntimeModelExecutionEngine` est crate-internal ; l'API de génération échoue s'il n'existe pas. |
| `ModelLoading` pilotable avec une décision de confiance externe | ✅ Fermé | `load_model` évalue la confiance depuis le `Runtime` scellé ; `ModelLoadingCoordinator::load` est crate-internal. |
| `LoadedModelContext` / residency proof forgeables | ✅ Fermé | Les champs d'autorité sont `pub(crate)` et ne peuvent plus être construits librement par un embedder externe. |
| `ModelInstance` marquée Ready avant les poids | ✅ Fermé | Création en `Loading`; passage à Ready après commit de matérialisation complet. |
| Matérialisation partielle non transactionnelle | ✅ Fermé | `WeightMaterializationTransaction` stage/commit/abort avec rollback des bindings et ressources déjà créés. |
| Poids vérifiés seulement par nom | ✅ Fermé | Les noms, digests de contenu et métadonnées shape/dtype du manifest sont propagés et vérifiés. |
| Fixture de poids en mémoire découplée de l'artifact réel | ✅ Fermé | Le chemin first-native matérialise les octets du vrai fixture Safetensors ; test de parité avec la représentation en mémoire. |
| Qwen Rust helper encore autoritaire en production | ✅ Fermé | `qwen_build_graph` / prefill / decode Rust sont `#[cfg(test)]`; le Component WASM est la source de sémantique de graphe de production. |
| Re-sélection Kernel après publication du plan | ✅ Fermé | `PreparedPlanNodeExecution` est consommé comme binding autoritaire ; le dispatch revalide sans recalculer une nouvelle sélection. |
| Exécution Reference CPU hors Provider path | ✅ Fermé pour le profil | Le chemin causal traverse le Provider enregistré et son `ProviderExecutionApi`. |
| Transport per-node privé en `HostTensor` | ✅ Fermé sur le chemin production | Les ressources intermédiaires passent via le contrat `TensorValue`; un guard de régression interdit le retour des appels HostTensor-typed dans la boucle de graphe de production. |
| KV recomputé comme oracle complet | ✅ Fermé sur le chemin production | KV par couche, staging pending puis promotion transactionnelle ; decode utilise le cache et la position absolue. |
| CLI dépendante du harnais E2E | ✅ Fermé | `run` / `chat` appellent le chemin Runtime first-native de production ; pas d'injection de logits ni d'appel Kernel/Reference CPU dans la CLI. |
| Faux sous-modules déclarés seulement dans `.gitmodules` | ✅ Fermé | Les six chemins sont de vrais gitlinks et disposent de jobs d'intégration dédiés. |
| GitHub Actions sur tags mutables | ✅ Fermé | Les actions utilisées dans `quality.yml` sont pinées sur des SHA immuables. |
| Signatures cryptographiques | ⏳ Déféré explicitement | #37, documenté dans `SECURITY.md`; ne pas présenter la supply chain actuelle comme authentifiée cryptographiquement. |
| Isolation model-family exhaustive | ⚠️ Partiel | #39 reste ouvert : la protection CI est réelle mais l'allowlist n'est pas exhaustive. |

## 4. Autorité Runtime et chaîne causale d'inférence

### 4.1 Logits produits par le modèle

Le contrat `RuntimeModelExecutor` est maintenant correctement orienté :

- `execute_prefill(instance, plan, input)` produit `PrefillExecutionResult.logits` ;
- `execute_decode_step(instance, plan, input)` produit `DecodeExecutionResult.logits` ;
- `ModelInstance` et `PreparedExecutionPlan` sont des préconditions explicites ;
- aucun tableau de logits n'est fourni par l'appelant à ce niveau.

C'est la correction structurante la plus importante du précédent audit : le sampling ne peut plus être alimenté par un faux calcul situé hors du datapath modèle.

### 4.2 Moteur d'exécution scellé

`RuntimeBuilder::model_execution_engine` et `Runtime::model_execution_engine` sont `pub(crate)`. Un consommateur externe du crate ne peut donc pas enregistrer arbitrairement un moteur qui fabriquerait des logits puis appeler le reste de l'API comme si le Runtime les avait calculés.

La génération côté `inference_api` résout le moteur détenu par le Runtime et retourne une erreur structurée si aucun moteur n'est présent. Le contrôle d'autorité est donc du bon côté de la frontière.

### 4.3 Plans préparés et dispatch

Le dispatch sait construire un `KernelDispatchPlan` depuis un `PreparedPlanNodeExecution`. Le commentaire et l'implémentation convergent : une fois le plan publié, le binding Kernel/Provider/Device est autoritaire ; le Runtime peut revalider l'état courant, mais ne doit pas relancer une sélection opportuniste qui ferait mentir la preuve du plan.

Le chemin first-native collecte également l'identité réelle de soumission / complétion issue du `ProviderExecutionApi`, ce qui donne une chaîne causale exploitable pour la conformance et l'observabilité.

## 5. Model Loading, provenance et matérialisation

### 5.1 Trust scellé dans le Runtime

`load_model` reçoit `&mut Runtime`, récupère `runtime.trust_store().evaluate(manifest)`, puis appelle la primitive crate-internal de chargement. Une décision de confiance évaluée par un `ModelTrustStore` créé par l'appelant n'est plus suffisante pour faire accepter un artifact à un Runtime qui ne l'a pas lui-même approuvé.

Le trust store du Runtime est configuré au build et n'est pas exposé ensuite comme objet mutable remplaçable.

### 5.2 Preuves de chargement non forgeables

Les structures qui constituent une preuve de chargement / résidence (`ModelLoadingResidencyPlan`, `LoadedModelContext` et champs associés) ont leurs données d'autorité en `pub(crate)`. L'API publique conserve les accesseurs nécessaires à l'observation mais plus la possibilité de fabriquer une fausse preuve depuis l'extérieur du crate.

### 5.3 Poids issus de l'artifact réel

Le chemin first-native ne se contente plus d'un `BTreeMap` de tensors construit en mémoire : les poids du fixture sont matérialisés depuis le vrai fichier Safetensors versionné. Le helper générique `host_tensors_from_artifact_bytes` :

- applique le `data_section_start` ;
- utilise des additions checked pour les offsets/taille ;
- vérifie la taille impliquée par la shape ;
- rejette explicitement les dtypes non supportés au lieu de réinterpréter les octets ;
- retourne des erreurs structurées en cas de dépassement ou incohérence.

Une régression réelle d'offset avait d'ailleurs été détectée par le test de parité artifact ↔ tensors, ce qui est un bon signal : le test prouve des octets, pas seulement des shapes.

### 5.4 Transaction de matérialisation et readiness

`materialize_model_instance_weights` ouvre un `WeightMaterializationTransaction`, stage chaque poids, abort au premier échec et commit seulement lorsque l'ensemble a réussi.

Le test de régression associé vérifie deux propriétés différentes :

1. l'instance qui échoue n'a jamais annoncé `Ready` ;
2. les poids déjà staged ne restent ni liés à l'instance ni présents dans le stockage Provider après rollback.

Les digests de contenu et shape/dtype déclarés sont également transportés depuis le manifest vers la définition de l'instance. Une mutation des octets est testée à la frontière publique de matérialisation et doit produire un mismatch de digest.

**Résultat : les deux P0 historiques sont fermés.**

## 6. Qwen Component et autorité du graphe

Le module Qwen Rust conserve des types/configurations utiles, mais les helpers historiques qui produisaient directement les graphes (`qwen_build_graph`, `qwen_prefill_graph`, `qwen_decode_graph`) sont désormais `#[cfg(test)]`.

Leur rôle est explicitement celui d'oracle de test / conformance. Le chemin de production obtient la sémantique de graphe depuis le véritable WASM Component. C'est conforme à l'objectif architectural :

- le Component décrit la sémantique portable ;
- le Runtime valide / planifie ;
- le Registry sélectionne ;
- le Provider exécute ;
- le Component ne choisit pas le Provider ni le Device.

Je ne retrouve pas de fallback de production vers l'ancien constructeur Rust qui rendrait le WASM Component décoratif.

## 7. Tensor Resources et KV incrémental

La boucle de production `execute_qwen_graph_nodes` a migré son transport inter-nœuds vers `TensorValue` et les ressources du Provider. Les matérialisations host restantes correspondent à des frontières explicites (poids, concaténation KV, extraction finale, résolution d'entrée lorsque le Provider courant exige du host), et non à un cache privé parallèle qui court-circuiterait les ressources.

Un guard de test inspecte la source de cette boucle pour empêcher la réintroduction silencieuse des anciennes méthodes `HostTensor`-typed du `ProviderExecutionApi`.

Le KV suit désormais une logique transactionnelle par couche : staging des nouvelles ressources, puis promotion atomique. Le decode conserve une position absolue explicite et réutilise les données historiques plutôt que de prétendre être incrémental tout en recalculant tout le préfixe.

Les fonctions « oracle » qui conservent encore des lectures/écritures HostTensor directes sont `#[cfg(test)]` et ne constituent pas le datapath de production.

## 8. CLI / Runtime boundary

Le découplage est maintenant crédible :

- `magnetar run` / `chat` délèguent la génération au Runtime ;
- la CLI n'importe pas les fonctions de Kernel Reference CPU pour calculer le modèle ;
- aucun fallback logits n'est présent dans le chemin normal ;
- fichiers, Git, workspace, réseau, outils et secrets restent côté CLI ;
- seul le contexte déjà matérialisé en texte traverse vers le Runtime ;
- `ModelRef` rejette les références ressemblant à des chemins ;
- les sessions de chat Runtime persistent réellement entre les tours.

### Observation P3 — commentaire chat template inexact

Il reste une incohérence documentaire dans `magnetar-cli/src/pipeline.rs` : le commentaire de `ChatSession::turn` affirme que, passé le premier tour, la CLI transmet des `PromptInput::ChatMessages` afin que **Runtime applique** le template autorisé. L'implémentation construit les `ChatMessage`, appelle `CliChatTemplateFormatter::format` **dans la CLI**, puis passe la chaîne rendue à `self.chat.turn`.

Ce comportement reste conforme à la spec, qui dit que Runtime **MAY** appliquer un template autorisé. Il faut donc simplement choisir l'une des deux vérités et l'écrire correctement :

- soit conserver le pré-rendu CLI et corriger le commentaire ;
- soit faire réellement traverser `PromptInput::ChatMessages` si l'objectif produit est de tester le templating Runtime.

**Sévérité : P3 / documentation, non bloquant.**

## 9. Sous-modules, formats et découplage d'extensions

Les six modules externes sont maintenant de vrais submodules Git :

- `formats/gguf`
- `formats/safetensors`
- `components/qwen`
- `components/llama`
- `providers/cpu`
- `providers/cuda`

La CI comporte des jobs distincts pour :

- Component integration ;
- Format integration ;
- Provider integration ;
- submodule integration et vérification de l'absence de dépendance directe du Core vers les modules externalisés.

Les parsers formats sont testés comme crates indépendants, tandis que le Runtime consomme un inventaire portable de tensors / octets. Cette direction est cohérente avec l'architecture d'extensions annoncée.

## 10. Sécurité et supply chain

### 10.1 Actions GitHub

Les `uses:` de `quality.yml` sont pinés à des SHA immuables avec le tag lisible conservé en commentaire. Le risque classique de tag mutable dans la CI a donc été corrigé.

### 10.2 WASM Component

`SECURITY.md` documente clairement le modèle : Component non fiable, Wasmtime sans WASI ambiant, fuel/epoch interruption et limites de ressources. Les Providers restent du code natif de confiance choisi par l'opérateur.

### 10.3 Signatures d'artifacts

La chaîne n'authentifie pas encore cryptographiquement le publisher. C'est explicitement documenté et tracé par #37. Le point important est que les métadonnées publisher/source seules ne deviennent pas une preuve : le Runtime exige toujours un mécanisme de confiance explicite (digest pinning / politique locale de développement).

Je classe donc #37 comme **gap de supply-chain connu**, pas comme bypass de la politique courante.

## 11. CI revalidée sur le HEAD

Le workflow **Quality #149** associé au SHA `e108a1d3910b89bd6381104dbaa857c13d86fb35` est terminé en **success**.

Les **23 jobs** observés sont verts :

- `quality / cargo-deny`
- `quality / wit`
- `quality / submodule integration`
- `quality / component integration`
- `quality / wasmtime component engine`
- `quality / coverage`
- `quality / rustfmt`
- `quality / msrv`
- `quality / check` sur Linux, Windows et macOS
- `quality / clippy`
- `quality / test` sur Linux, Windows et macOS
- `quality / provider conformance`
- `quality / e2e conformance`
- `quality / provider integration`
- `quality / openspec`
- `quality / model-family isolation`
- `quality / docs`
- `quality / wasm32 component engine`
- `quality / format integration`

Ce résultat ne remplace pas la revue d'architecture, mais il apporte une preuve importante : les garde-fous ajoutés ne sont pas seulement présents dans les YAML, ils s'exécutent avec succès sur le HEAD audité.

## 12. Résidu P1/P2 revalidé

### #38 — multi-output graph propagation — P1, non bloquant pour le baseline actuel

Le constat est toujours vrai : `dispatch_reference_cpu_operator` reçoit un unique tuple `output`, alors que `KernelResult.updated_resources` sait représenter plusieurs sorties. Le baseline Qwen actuel est mono-sortie, donc ce défaut n'affecte pas le profil revendiqué aujourd'hui.

**Gate recommandé :** à fermer avant l'introduction d'un Component utilisant réellement un opérateur multi-sortie dans ce datapath.

### #39 — model-family isolation — P1, dette d'architecture

Le job CI protège une liste explicite de modules Core, mais pas tous les modules génériques. `conformance.rs` contient par exemple encore `QwenWasmModelComponent` dans une enum de profil générique.

**Gate recommandé :** à fermer avant d'annoncer le Core comme totalement family-agnostic ou d'ajouter plusieurs familles natives au même niveau de maturité.

### #40 — source `DevelopmentFixture` vs `ClientProvided` — P1, provenance

Le chemin `register_qwen_component_artifact` reçoit des octets fournis par l'embedder mais construit encore `ComponentDistributionSourceKind::DevelopmentFixture`.

La correction est petite mais sémantiquement importante. Elle n'est pas un trust bypass aujourd'hui parce que la source déclarative ne suffit pas à rendre l'artifact trusted.

**Gate recommandé :** corriger avant de s'appuyer sur la provenance pour diagnostics, policy ou audit externe.

### #41 — erreurs `write_tensor` / `release_tensor` — P1, fiabilité Provider

La transaction de poids est correcte dans son ordre admission → write → commit/rollback, mais le contrat Provider ne permet pas encore à une implémentation de signaler proprement un échec de write ou de release.

**Gate recommandé :** à fermer avant de qualifier comme robuste un Provider matériel / distant / faillible, notamment CUDA réel.

### #42 — instance bloquée en `Loading` — P2, lifecycle

Le nouvel état `Loading` est désormais réellement observable, mais une instance abandonnée à ce stade n'a pas encore de chemin explicite de cancellation/unload.

**Gate recommandé :** à fermer avant d'exposer des workflows de chargement longs, asynchrones ou interrompables.

### #37 — signature cryptographique — P2, supply chain

Le digest pinning protège l'intégrité contre la substitution non autorisée lorsqu'il est configuré, mais il ne prouve pas l'identité d'un éditeur.

**Gate recommandé :** nécessaire avant toute promesse de chaîne de distribution signée / publisher-authenticated.

## 13. Topologie Git de la PR

La comparaison Git signale que la branche est « diverged » de `main` : 71 commits ahead et 2 commits behind, avec merge-base `09df99ec…`.

Ce point n'est **pas** une divergence de contenu problématique. `09df99ec…` est le HEAD de la PR #35 et `dbf4551a…` son merge commit sur `main`; la comparaison entre ces deux commits ne contient aucune différence de fichiers. Les deux commits « behind » sont donc liés à l'historique du merge, pas à du code manquant.

Aucun rebase n'est requis pour corriger un écart fonctionnel identifié par cet audit. Un squash/rebase peut rester un choix d'hygiène d'historique, pas un gate technique.

## 14. Décision de gate

### Merge de la PR #36

**APPROVE.**

Conditions vérifiées :

- pas de P0 résiduel identifié ;
- chaîne causale first-native crédible de l'artifact jusqu'aux logits ;
- Qwen Component réellement autoritaire en production ;
- Runtime propriétaire des plans, de l'instance, du KV et du Provider dispatch ;
- matérialisation des poids réelle, vérifiée et transactionnelle ;
- CLI sans bypass de calcul modèle ;
- OpenSpec strict vert ;
- tests multi-plateformes et conformance verts ;
- supply-chain CI renforcée par SHA pinning.

### Ce que ce verdict ne signifie pas

Ce verdict **ne signifie pas** que Magnetar est déjà :

- un runtime général multi-familles totalement isolé ;
- prêt pour des Providers matériels arbitrairement faillibles ;
- prêt pour des Components multi-sorties sur ce datapath ;
- doté d'une supply-chain d'artifacts cryptographiquement authentifiée ;
- exempt de dette lifecycle sur les chargements interrompus.

Ces limites sont correctement représentées par #37 à #42 et doivent rester des gates de la roadmap correspondante.

## 15. Actions recommandées après merge

1. **#40 en premier** : correction courte, forte valeur sémantique sur la provenance.
2. **#41 avant CUDA réel** : transformer `write_tensor` / `release_tensor` en `Result` et injecter des échecs dans les tests transactionnels.
3. **#38 avant premier opérateur multi-sortie** : supprimer l'hypothèse single-output de l'exécuteur générique.
4. **#39 avant deuxième famille au même niveau** : rendre le guard family-isolation structurel/exhaustif.
5. **#42 avant loading asynchrone/long** : introduire cancellation/unload depuis `Loading`.
6. **#37 avant promesse supply-chain forte** : définir puis implémenter la signature et la révocation.
7. Corriger le commentaire de `ChatSession::turn` ou faire réellement traverser `PromptInput::ChatMessages` selon l'architecture souhaitée.

---

**Verdict final : `APPROVE` pour le first-native baseline borné de la PR #36, avec dette P1/P2 explicitement tracée et sans nouveau blocker détecté.**
