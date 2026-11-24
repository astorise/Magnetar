# RFC — Magnetar : écosystème de kernels spécialisés et d'Accelerator Packages

> **Statut :** proposition d'architecture à discuter (non implémentée de bout en bout)  
> **Date :** 23 novembre 2026  
> **Périmètre :** Magnetar Runtime, Components WASM, Kernel Registry, Providers CPU/CUDA/ROCm/Metal/WebGPU/OpenVINO/QNN, distribution OCI  
> **Décision proposée :** **GO pour un POC ciblé**, sous réserve de qualification, de sécurité et de mesures de performance.  
> **Nature du document :** synthèse et formalisation des échanges ; les interfaces et manifestes marqués « proposition » ne sont pas des API existantes.

## 1. Synthèse exécutive

Magnetar pourrait améliorer la vitesse d'inférence en permettant à des **Components WASM de proposer des implémentations spécialisées d'opérateurs**, distribuées et mises à jour indépendamment des modèles et des Providers. L'exécution resterait intégralement sous le contrôle du Runtime et du Provider matériel.

La proposition ne consiste **pas** à autoriser un Component à invoquer CUDA ou à enregistrer directement un kernel. Elle consiste à introduire un **canal de fourniture d'artefacts** — potentiellement un Component WASM qui sert des octets — raccordé aux contrats existants : **Kernel Exchange Bundle → ingestion → validation et qualification → préparation par le Provider → Kernel Registry → sélection et dispatch**.

Le *hot loading* permettrait de préparer une nouvelle génération pendant que les inférences continuent avec l'ancienne. Le *hot swap* n'interviendrait qu'après validation et promotion explicite, puis libération de l'ancienne génération lorsque toutes ses références seraient terminées.

À moyen terme, il faut généraliser la notion de « kernel » à plusieurs types d'**artefacts d'accélération** : `OperatorKernel`, `FusedKernel`, `CompiledGraph`. **Le canal de distribution peut être unifié ; les mécanismes d'exécution, eux, doivent rester propres à chaque Provider.**

### Principes non négociables

1. **Le Component décrit des sémantiques et propose des artefacts ; il ne choisit ni Provider ni Device et ne détient aucun pointeur natif.**
2. **Le Runtime demeure l'autorité** sur l'admission, les contraintes, les ressources, la sélection, la promotion, le rollback et la traçabilité.
3. **Le Provider reste propriétaire** de la compilation éventuelle, du chargement, des handles natifs, de l'exécution et de la libération.
4. **Aucune compilation ni qualification bloquante dans le chemin chaud de décodage des tokens.**
5. **Un digest prouve l'identité/intégrité des octets, pas leur innocuité ni une autorisation d'exécution.**
6. **Un kernel spécialisé ne remplace un kernel connu comme correct que s'il est admissible, qualifié et utile dans un contexte mesuré.**

## 2. État de référence dans Magnetar

Les constats suivants proviennent de l'inspection des dépôts au cours de l'échange ; ils ne constituent pas un nouvel audit exhaustif des tests et performances.

| Sous-système | Constat | Conséquence |
|---|---|---|
| `magnetar-kernel` | Contrats de `Kernel Registry`, dispatch, artefacts source/compilés/préparés, cache, ingestion, qualification, autotuning, benchmarks et générations | Réutiliser les abstractions existantes plutôt que créer un second registre |
| `Kernel Registry` | Les enregistrements directs des `Component`/clients sont refusés ; l'autorité appartient aux Providers et aux fixtures Runtime | Conserver l'invariant et faire passer les propositions par une voie contrôlée |
| `Kernel Exchange Bundle` | Manifest JSON canonique `magnetar:kernel-manifest@1.0`, blobs adressés par SHA-256, support de variantes | Bon format de référence pour les kernels classiques |
| Distribution | OCI/GHCR déjà prévu : `application/vnd.magnetar.kernel-bundle.v1+tar+gzip` | Ne pas créer un format de transport concurrent |
| Provider CUDA | Exécution réelle ; compilation NVRTC interne ; kernels optimisés, résidence device et capture/replay CUDA Graphs | Candidat naturel au premier POC |
| API générique de compilation Provider | `kernel_compilation_capability()` / `kernel_compilation_api()` sont prévues et facultatives ; pas exposées par l'implémentation CUDA inspectée | Ajouter l'adaptateur de préparation/compilation aux kernels externes |
| Provider CPU | Référence numérique fonctionnelle et détection de fonctions SIMD | Excellent oracle et second POC de chargement spécialisé |
| Provider ROCm | Découverte matérielle, pas encore de calculs qualifiés dans le dépôt inspecté | Reporter l'intégration après un chemin d'exécution réel |
| Provider Metal | Placeholder, sans calcul natif vérifié dans le dépôt inspecté | Reporter après accès à du matériel Apple et validation |
| Provider QNN | Découverte des backends Qualcomm, pas encore de calcul ni qualification sur matériel réel | Faire de la spécialisation de graphe/op-package un chantier ultérieur |
| Dépôt nommé `Magnetar-provider-OpenVINO` | Le README décrit en réalité une découverte NPU via **oneAPI Level Zero**, et non encore un backend OpenVINO opérationnel | Clarifier la frontière et la stratégie Intel avant toute API spécialisée |

**Attention à la documentation historique :** certains README décrivent une baseline plus ancienne que le code courant. Pour CUDA, le code `executor.rs` décrit notamment la conservation des tenseurs sur le GPU entre invocations. Vérifier code, tests et révision Git avant d'affirmer qu'une limite historique existe toujours.

**Sources principales :** [Kernel Registry et Dispatch](https://github.com/astorise/Magnetar/blob/main/docs/architecture/kernel-registry-dispatch.md), [contrat de Model Component](https://github.com/astorise/Magnetar/blob/main/docs/architecture/model-component.md), [artefacts kernels](https://github.com/astorise/Magnetar/blob/main/openspec/specs/kernel-artifact/spec.md), [ingestion](https://github.com/astorise/Magnetar/blob/main/openspec/specs/kernel-artifact-ingestion/spec.md), [distribution](https://github.com/astorise/Magnetar/blob/main/openspec/specs/component-artifact-distribution/spec.md), [Provider CUDA](https://github.com/astorise/Magnetar-provider-CUDA), [Provider QNN](https://github.com/astorise/Magnetar-provider-qnn), [Provider OpenVINO nommé](https://github.com/astorise/Magnetar-provider-OpenVINO).

## 3. Motivations et gains attendus

Les optimisations les plus prometteuses portent sur des calculs dépendant de l'architecture du modèle, du dtype, du layout, de la forme des tenseurs et des ressources matérielles :

- **Attention spécialisée** (variantes GQA/MLA, dimensions de tête et séquences particulières) ;
- **Fusions d'opérateurs** compatibles (par exemple normalisation + projection, activation + multiplication), afin de réduire lancements et trafic mémoire ;
- **MoE** : routage, regroupement des tokens par expert, GEMM groupées ;
- **Quantification** : déquantification fusionnée avec MatMul, formats et tailles de blocs spécifiques ;
- **Spécialisations par charge** : prefill versus decode, batch, contexte, topologie GPU et jeu d'instructions CPU.

**Les gains ne sont pas garantis.** Un kernel spécialisé peut être moins performant qu'un kernel existant déjà optimisé. L'évaluation doit mesurer le coût **de bout en bout**, y compris préparation, transferts et conversions explicites, mémoire supplémentaire et interactions avec le scheduler.

## 4. Architecture cible

```text
  Model Component WASM                  Kernel Package Component WASM
  (Qwen / Llama / Gemma...)             (facultatif, mutualisable)
      |                                     |
      | Graphe / opérateurs portables       | Catalogue / manifeste / blobs
      | + références ou recommandations     | (aucun handle GPU)
      +------------------+------------------+
                         |
              Magnetar Runtime
         +---------------+--------------------+
         | Kernel Artifact Source Gateway     |
         | - Sources autorisées : WASM / OCI   |
         | - Lecture bornée et par digest      |
         +---------------+--------------------+
                         |
         +---------------v--------------------+
         | Ingestion Gateway / Artifact Cache |
         | Contrôles, signatures, politique   |
         | Validation sémantique, quarantaine |
         +---------------+--------------------+
                         |
         +---------------v--------------------+
         | Qualification / Benchmarks /       |
         | Autotuning / Promotion Policy      |
         +---------------+--------------------+
                         |
              Kernel Registry / Dispatch
                         |
         +---------------v--------------------+
         | Provider matériel sélectionné      |
         | CPU / CUDA / ROCm / WGPU / ...     |
         | Prépare, charge, exécute, libère    |
         +------------------------------------+
```

Le lien **Component → kernel** est **non autoritaire** : il référence des sémantiques d'opérateurs et, éventuellement, recommande des artefacts. La décision exécutable est prise uniquement après admission et sélection Runtime. Aucun nouveau chemin d'invocation ne doit court-circuiter le Kernel Registry.

## 5. Format de distribution : réutiliser Kernel Exchange Bundle

### 5.1. Ce qui existe déjà

Le format existant offre :

- un fichier `kernel.manifest.json` sous schéma `magnetar:kernel-manifest@1.0` ;
- des blobs contenus dans `blobs/sha256/<digest>` ;
- des identifiants de format extensibles : par exemple `nvidia:ptx`, `nvidia:cubin`, WGSL, ou formats futurs ;
- des métadonnées de compatibilité : opérateur(s), variantes et architecture ciblée, dtype, forme/layout, précision, provenance, références de preuves ;
- un transport OCI (`tar+gzip` dans le profil de publication actuel) sans faire du chemin réseau ou du tag une autorité de confiance.

**Proposition :** maintenir ce format comme source de vérité pour `OperatorKernel` et `FusedKernel`. Le transport peut être un bundle embarqué dans un WASM, un fichier local, un blob du cache ou un artefact OCI ; l'identité logique demeure fondée sur le contenu validé. Ne pas confondre **digest du manifeste**, **digests des blobs** et **digest de l'artefact OCI** : ils identifient des objets distincts.

### 5.2. Deux modes de livraison

| Mode | Fonctionnement | Avantages | Contraintes |
|---|---|---|---|
| **A — WASM autonome** | Le Component embarque les bundles et expose leurs octets via WIT | Hors ligne, installation simple, packaging autonome | Taille du WASM, duplication entre Components, mise à jour plus lourde |
| **B — WASM léger + OCI/cache** | Le Component expose un catalogue et des digests ; le Runtime résout les bundles depuis une source explicitement autorisée | Mutualisation, cache, mises à jour indépendantes, diffusion à grande échelle | Disponibilité de la source, authentification, résolution hors ligne, provenance |

**Recommandation : mode B par défaut ; mode A pour installations hors ligne ou embarquées.** La provenance et l'autorisation ne découlent jamais du simple fait qu'un Component sert des octets.

### 5.3. Extension future : Accelerator Package

Pour les Providers qui exécutent des **graphes compilés** plutôt que des kernels isolés, étudier une enveloppe/version complémentaire, sans casser la compatibilité `Kernel Exchange Bundle` v1.

Trois catégories proposées :

- `OperatorKernel` : implémentation native d'un opérateur portable (CPU, CUDA, ROCm, WGSL, Metal).
- `FusedKernel` : implémentation d'une composition sémantique explicite d'opérateurs.
- `CompiledGraph` : graphe ou contexte exécutable cible (OpenVINO NPU, QNN HTP notamment), avec compatibilité plus restrictive et substitution au niveau d'un segment ou d'une instance.

Exemple **conceptuel, non conforme à un schéma actuellement implémenté** :

```yaml
schema: magnetar:accelerator-package@1  # PROPOSITION
name: optimized-attention
version: 1.0.0
semantic_binding:
  operator: magnetar:operator/attention
  version: 1
variants:
  - id: cpu-avx512
    kind: operator-kernel
    provider: cpu
    format: native-elf
    requirements: [avx512f]
    artifact: sha256:<digest-cpu>
  - id: cuda-sm90
    kind: operator-kernel
    provider: cuda
    format: nvidia:cubin
    architecture: sm90
    artifact: sha256:<digest-cuda>
  - id: openvino-npu
    kind: compiled-graph
    provider: openvino
    format: openvino:compiled-model
    artifact: sha256:<digest-openvino>
  - id: qnn-htp
    kind: compiled-graph
    provider: qnn
    format: qnn:context-binary
    artifact: sha256:<digest-qnn>
```

Les variantes NPU devront encoder notamment runtime/SDK, version du compilateur, génération d'accélérateur, forme, layout, contraintes de graphe, et éventuellement un **lien strict au modèle, aux poids ou à leur hash**. Un graphe compilé QNN ou OpenVINO n'est pas nécessairement réutilisable comme un kernel d'attention générique.

## 6. Contrat WIT : un Component peut servir des binaires

Créer un rôle facultatif `kernel-package` et une **nouvelle interface de source d'artefacts**, sans nouvelle autorité GPU. Exemple **proposé** :

```wit
package magnetar:kernel-source@0.1.0;

interface source {
  record bundle-descriptor {
    id: string,
    manifest-digest: string,
    size: u64,
  }

  list-bundles: func() -> list<bundle-descriptor>;

  read-chunk: func(
    manifest-digest: string,
    offset: u64,
    max-bytes: u32,
  ) -> result<list<u8>, string>;
}

world kernel-package {
  export source;
}
```

**Précisions à traiter avant stabilisation :**

- Le `manifest-digest` doit désigner explicitement le manifeste ; il ne remplace pas l'identifiant du bundle complet. Le protocole peut évoluer vers des identifiants de blobs distincts et un `open-artifact` opaque.
- Le Runtime impose taille maximale par fragment, quotas cumulés, durée, mémoire, vérification progressive des octets et absence de lecture hors limites. Ne jamais faire confiance au `size` annoncé sans contrôle.
- La méthode permet de **fournir des octets**, pas de déclencher `cuModuleLoad*`, `dlopen`, `hipModuleLoadData` ou de récupérer un pointeur de tenseur.
- Dans le mode OCI, le Component ne bénéficie pas d'un accès réseau ambiant : l'**Artifact Source Gateway côté Runtime** résout uniquement des localisateurs autorisés.
- Une extension asynchrone / streaming pourra être évaluée ultérieurement selon les contraintes du moteur WASM retenu.
- Le chargement doit être effectué hors du chemin chaud d'inférence.

## 7. Hot loading, promotion, rollback et retrait

Distinguer impérativement : **hot delivery** (nouveaux octets disponibles), **hot preparation** (nouveau module prêt), **hot activation** (nouvelle génération sélectionnée). Une réception réussie **n'implique jamais** exécution.

Séquence cible :

1. **Discover** — un Component annonce un bundle et ses propriétés.
2. **Resolve / Stage** — le Gateway lit ou récupère des octets dans une transaction isolée.
3. **Validate / Ingest** — contrôles de schéma, digest, signature, provenance, politique ; acceptation, quarantaine ou rejet.
4. **Qualify** — tests numériques et sémantiques sur la cible et sur les spécialisations effectivement utilisées.
5. **Prepare** — le Provider charge la variante cible ; compilation éventuelle dans le chemin froid.
6. **Benchmark** — comparaison contrôlée à un kernel de référence, distincte de la qualification.
7. **Ready** — la génération est admissible mais pas nécessairement active.
8. **Promote atomically** — le Runtime publie la nouvelle génération à une frontière sûre de plan/invocation ; les nouvelles invocations la sélectionnent selon leur politique.
9. **Drain / Retire** — les travaux et captures encore attachés à l'ancienne génération se terminent avant destruction des handles natifs.
10. **Rollback** — en cas de dégradation ou d'échec, retour explicite à une génération connue comme correcte.

Pour CUDA, la préparation peut s'appuyer sur le chargement d'un module PTX/CUBIN depuis la mémoire par le Driver API. **Attention :** les CUDA Graphs capturés, flux, événements et fonctions encore référencés imposent une gestion de durée de vie robuste ; ne pas décharger le module tant qu'ils sont actifs. La même exigence s'applique aux contextes CPU/JIT, graphes OpenVINO ou QNN, avec des règles spécifiques.

Une stratégie réaliste est **generation pinning** (référence opaque + comptage de références / quiescence) et promotion à la frontière d'un plan préparé ; pas de mutation d'un graphe déjà en cours.

## 8. Spécialisation et exécution par Provider

| Provider | Unité naturelle d'accélération | Formats ou mécanismes envisagés | Hot activation | Priorité proposée |
|---|---|---|---|---|
| **CPU** | Opérateur/fusion SIMD | Code machine AOT signé, variantes AVX2/AVX-512/AMX/NEON/SVE ; JIT optionnel | Possible via génération de fonction/module à ABI maîtrisée | **P1** |
| **CUDA** | Kernel/fusion/variante matérielle | PTX, CUBIN, compilation NVRTC hors chemin chaud | Favorable, via module et fonctions préparés | **P0** |
| **ROCm** | Kernel/fusion | HSACO, HIP/hipRTC selon support du Provider | Possible après implémentation d'exécution | **P2** |
| **Metal** | Shader/pipeline de calcul | MSL, metallib, pipelines Metal | Possible après implémentation native | **P2** |
| **WebGPU** | Pipeline de calcul | WGSL puis pipeline préparé | Possible, soumis au backend et aux contraintes WebGPU | **P2** |
| **OpenVINO CPU/GPU** | Extension/opérateur ou sous-graphe | Extensions C++, opérations custom, compilations spécifiques | Variable ; peut nécessiter de nouvelles instances | **P3** |
| **OpenVINO NPU** | Graphe compilé cible | Modèle compilé compatible pilote/runtime | Contraint : swap de graphe/instance plutôt que kernel brut | **P3** |
| **QNN CPU/GPU/HTP** | Op Package ou contexte QNN | Op Packages, graphes et contextes compilés QAIRT/QNN | Contraint : nouveaux contextes/graphes, versions SDK | **P3** |

**Portabilité :** partager les **sémantiques** et le **contrat de sélection**, pas prétendre qu'un même binaire s'exécute partout. Un artefact `CompiledGraph` ne doit pas être injecté comme un `OperatorKernel` : les unités d'admission, de résidence mémoire et de remplacement diffèrent.

**Intel :** distinguer l'API OpenVINO de la couche oneAPI Level Zero. Le dépôt portant le nom OpenVINO n'implémente pas aujourd'hui un Provider OpenVINO de calcul ; ce point doit faire l'objet d'un arbitrage explicite.

## 9. Politique de sélection et optimisation

Le `Kernel Registry` doit filtrer les implémentations **éligibles avant tout classement par performance**. Critères :

- identité et version de la sémantique d'opérateur, ordre de fusion, attributs ;
- Provider/Device réellement disponibles et compatibles ;
- architecture cible, dtype, layout, forme, batch, préfill/decode, précision et déterminisme ;
- allocation/résidence, contraintes d'affinité, workspace et budget mémoire ;
- preuves de qualification valides et politique de confiance ;
- coût complet : exécution, transferts explicites, conversions, latence, mémoire, préparation amortie ;
- stabilité du résultat : benchmarks pertinents, hystérésis et contrôle de régressions.

Un Component peut **recommander**, mais ne peut pas forcer l'exécution d'un kernel ni contourner les contraintes de données et d'affinité. Le fallback est **explicite** vers une autre variante, un autre Provider avec plan de mouvement valide, ou une erreur structurée.

## 10. Sécurité et isolation

Le fait que les octets transitent via un WASM sandboxé **ne rend pas leur exécution native sandboxée**. Un CUBIN/PTX, une bibliothèque CPU, un Op Package QNN ou une extension C++ peut exécuter du code non fiable avec un impact hors du contexte WASM. Le POC de production doit donc exiger :

1. **Politique de confiance distincte** : éditeurs autorisés, signatures/provenance, empreintes, révocations, et permissions d'exécution propres au déploiement. La présence dans OCI ou en cache ne suffit pas.
2. **Ingestion transactionnelle** : limites de taille/décompression, anti-traversée de chemin et anti-symlinks, intégrité des octets exacts, absence d'effet de bord de validation.
3. **Pas d'autorité ambiante** : ni filesystem, ni réseau, ni processus, ni `dlopen` déclenchés librement par le Component ; aucune URL arbitraire ou commande shell empruntée à un manifeste.
4. **Qualification obligatoire** avant activation ; exécution des tests de code natif non fiable dans un environnement de test isolé et explicitement autorisé lorsque possible.
5. **Compatibilité et ABI versionnées**, limitation des ressources, diagnostics sans exfiltration de prompts/poids/adresses mémoire, audit des transitions.
6. **Retrait sûr et rollback** avec conservation des générations actives ; politique de désactivation des packages révoqués.

**Limite explicite :** ni la signature ni les tests numériques ne prouvent l'innocuité d'un kernel natif. Une politique de confiance du code exécutable reste nécessaire, et un processus séparé ne garantit pas à lui seul une isolation matérielle des kernels GPU.

## 11. Qualification et mesures de performance

Toute optimisation doit être évaluée avec le **même modèle, poids, prompt, précision acceptable, batch, contexte, matériel, pilote et politique mémoire**.

| Niveau | Mesures essentielles |
|---|---|
| Kernel | Latence GPU/CPU, occupation/mémoire, bande passante, lancement, workspace |
| Sous-graphe | Temps agrégé, nombre de lancements, transferts et conversions, mémoire intermédiaire |
| Inférence | TTFT, TPOT, tokens/s, p50/p95, consommation RAM/VRAM, qualité des sorties |
| Exploitation | Temps de préparation, taille des bundles, fréquence des swaps, taux de rollback, erreurs et pression mémoire |

Comparer **prefill et decode séparément** ; inclure plusieurs tailles de contexte et batch. Corriger les mesures de l'effet de warmup/cache. La qualification numérique (logits/tokens, tolérances explicites) est **indépendante** du benchmark de vitesse.

**Critère de promotion proposé :** gain reproductible sur une cible pertinente **sans régression de correction, de sécurité, de mémoire ou de latence de service**, avec seuil de gain fixé par la politique de l'équipe et supérieur au bruit expérimental. Aucun multiplicateur de performance ne peut être annoncé à ce stade.

## 12. Plan d'implémentation conseillé

### Lot 0 — RFC et contrats (Magnetar)

- Valider la terminologie : `Kernel Exchange Bundle` existant, `Kernel Artifact Source`, rôle `kernel-package`, puis éventuel `Accelerator Package` pour graphes compilés.
- Formaliser `Component → source d'artefacts` et compatibilité des versions WIT.
- Documenter la séparation des autorités, les frontières de confiance et le cycle de vie hot loading/hot swap.
- Réutiliser `magnetar-kernel` sans étendre l'autorité du Component sur le registre.

**Sortie :** OpenSpec approuvé, scénarios acceptation/rejet et tests de contrat.

### Lot 1 — POC CUDA minimal (Magnetar + Magnetar-provider-CUDA)

- Partir d'un **CUBIN précompilé** couvrant un opérateur portable simple ou une fusion déjà sémantiquement reconnue.
- Ajouter la préparation/chargement externe dans le Provider CUDA ; ne pas confondre compilation de source et chargement de binaire déjà compilé.
- Passer par ingestion, cache, qualification sur vrai GPU, publication comme candidat, sélection Runtime et dispatch réel.
- Conserver le kernel de référence et mesurer le coût de bout en bout.

**Sortie :** première exécution matérielle vérifiée d'un kernel externe via le chemin normal du Runtime.

### Lot 2 — WASM Artifact Source et hot swap

- Implémenter le Component `kernel-package`, lecture bornée des bundles intégrés et, séparément, résolution OCI autorisée.
- Raccorder au Gateway existant sans autoriser l'enregistrement direct de kernels.
- Préparer une génération N+1 en parallèle des inférences N ; promotion atomique, invalidation/recapture des plans et graphes impactés, retrait sûr, rollback.
- Ajouter des tests adversariaux : corruption, mauvais digest, mauvaise architecture, refus de confiance, charge concurrente, annulation, révocation, fuite de handles.

**Sortie :** hot swap sans redémarrage et sans interruption des inférences déjà engagées.

### Lot 3 — CPU et sélection multi-variantes

- Définir un ABI d'invocation CPU stable pour opérations spécialisées et la politique d'autorisation de code natif.
- Ajouter au moins deux variantes conditionnelles (ex. baseline et AVX2/AVX-512), avec fallback testé.
- Comparer coûts de calcul et mouvements de données sur une charge réelle.

**Sortie :** preuve que le protocole est réellement indépendant de CUDA.

### Lot 4 — Backends additionnels et graphes compilés

- Étendre successivement à WebGPU/ROCm/Metal lorsque les chemins de calcul sont effectivement vérifiés.
- Proposer le contrat `CompiledGraph`, avec **préparation et swap à la granularité segment/instance**, pour OpenVINO et QNN.
- Exiger SDK compatibles, matériel réel et qualification multi-cibles avant déclaration de support.

**Sortie :** matrice de support par Provider, type d'artefact et capacité de remplacement.

### Lot 5 — Écosystème (optionnel)

- Bibliothèque de packages d'accélération mutualisés, indépendants des Components de modèle.
- Publication OCI avec signatures et preuves de qualification/benchmarks par matériel.
- Optimiseur externe proposant de nouvelles variantes ; Magnetar conserve le dernier mot sur leur promotion.
- Découverte et recommandation de packages sans installation ou exécution implicite de code natif non autorisé.

## 13. Critères d'acceptation de la première expérimentation

- [ ] Un `Component WASM` fournit un bundle depuis ses octets embarqués, sans permission GPU ni système.
- [ ] Le même bundle peut être importé via le chemin OCI ou local autorisé, avec identité logique vérifiée.
- [ ] Un bundle corrompu, mal ciblé, non qualifié ou non autorisé est rejeté/quarantainé et n'atteint jamais le dispatch.
- [ ] Un Provider CUDA prépare un CUBIN externe et annonce le kernel selon les règles du registre existant.
- [ ] Le kernel qualifié est exécuté sur GPU réel avec résultats comparés au CPU de référence.
- [ ] Une génération N+1 devient active sans redémarrer Magnetar ; les invocations déjà associées à N se terminent correctement.
- [ ] Les captures CUDA Graphs, caches et plans obsolètes sont gérés sans libération anticipée.
- [ ] Rollback et révocation sont démontrés ; aucune compilation n'intervient dans la boucle de décodage.
- [ ] TTFT, TPOT, tokens/s, p50/p95 et mémoire sont mesurés avant/après sur un modèle et un matériel explicitement documentés.

## 14. Questions ouvertes à arbitrer avec l'équipe

1. **Granularité de publication :** un bundle par implémentation, ou collection multi-variantes sous une identité logique ?
2. **Interface WIT :** lecture par blocs (MVP) ou ressource/streaming asynchrone ; mécanisme précis d'identité `manifest`/`blob`/`bundle` ?
3. **Association aux Components de modèle :** manifeste statique, recommandation d'artefact lors du chargement, ou catalogue indépendant résolu par la politique Runtime ?
4. **Politique de confiance :** qui signe, qui autorise, comment révoquer et comment traiter les binaires embarqués dans un WASM signé ?
5. **Hot swap :** frontière d'activation (plan, invocation, instance) et traitement des CUDA Graphs/caches capturés ?
6. **ABI CPU :** natif signé AOT seulement pour le MVP, ou isolation/JIT distincts ?
7. **CompiledGraph :** extension de manifest v1 ou nouveau schéma, et métadonnées de liaison au modèle/poids ?
8. **Intel :** véritable Provider OpenVINO séparé d'un backend Level Zero, ou une stratégie NPU unifiée clairement délimitée ?
9. **Benchmarks :** charge de référence et seuils de gain pour autoriser automatiquement une promotion ?

## 15. Décision recommandée

**GO pour une preuve de faisabilité progressive**, pas pour l'activation immédiate de code natif arbitraire livré par WASM.

Commencer par le **chemin d'ingestion existant + Provider CUDA + CUBIN précompilé**, puis ajouter le **WIT Artifact Source** et le **hot swap générationnel**. Prouver ensuite la portabilité du concept sur CPU. Généraliser seulement après cela aux graphes compilés OpenVINO/QNN et aux autres accélérateurs.

**Vision cible :** un écosystème où des équipes publient des accélérations spécialisées indépendamment des modèles, où les Components WASM peuvent les découvrir ou les servir, et où Magnetar sélectionne, mesure, active et remplace les meilleures variantes admissibles **sans changer le code du modèle ni redémarrer le Runtime**.

---

## Références du projet

- [Magnetar](https://github.com/astorise/Magnetar)
- [Kernel Registry et Dispatch](https://github.com/astorise/Magnetar/blob/main/docs/architecture/kernel-registry-dispatch.md)
- [Model Component Contract](https://github.com/astorise/Magnetar/blob/main/docs/architecture/model-component.md)
- [Kernel Artifact Spec](https://github.com/astorise/Magnetar/blob/main/openspec/specs/kernel-artifact/spec.md)
- [Kernel Artifact Manifest Spec](https://github.com/astorise/Magnetar/blob/main/openspec/specs/kernel-artifact-manifest/spec.md)
- [Kernel Artifact Ingestion Spec](https://github.com/astorise/Magnetar/blob/main/openspec/specs/kernel-artifact-ingestion/spec.md)
- [Kernel Distribution Spec](https://github.com/astorise/Magnetar/blob/main/openspec/specs/kernel-distribution/spec.md)
- [Component Artifact Distribution Spec](https://github.com/astorise/Magnetar/blob/main/openspec/specs/component-artifact-distribution/spec.md)
- [Kernel Compilation Contract](https://github.com/astorise/Magnetar/blob/main/magnetar-kernel/src/kernel_compilation.rs)
- [Magnetar-provider-CUDA](https://github.com/astorise/Magnetar-provider-CUDA)
- [Magnetar-provider-CPU](https://github.com/astorise/Magnetar-provider-CPU)
- [Magnetar-provider-ROCm](https://github.com/astorise/Magnetar-provider-ROCm)
- [Magnetar-provider-Metal](https://github.com/astorise/Magnetar-provider-Metal)
- [Magnetar-provider-OpenVINO](https://github.com/astorise/Magnetar-provider-OpenVINO)
- [Magnetar-provider-qnn](https://github.com/astorise/Magnetar-provider-qnn)
