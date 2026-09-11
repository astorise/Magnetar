# Audit Magnetar — Production Qwen Model Loading

**Projet :** `astorise/Magnetar`  
**Date de l'audit :** 2026-09-11  
**Branche auditée :** `main`  
**SHA audité :** `9d57ad4a064dfbfedccbe7360a2216c36b3e19ee`  
**Change concerné :** `implement-production-qwen-model-loading`  
**Verdict global :** **CHANGES REQUIRED — NO-GO de clôture**

---

## 1. Résumé exécutif

L'implémentation de **Production Qwen Model Loading** est désormais substantiellement réelle et conforme à l'architecture cible de Magnetar.

Le chemin de production n'est plus une simple fixture : Magnetar sait maintenant ingérer un bundle Hugging Face Qwen réel, construire un `ModelInstance`, générer un graphe Qwen configurable, matérialiser les poids via les contrats Runtime, puis exécuter le modèle au travers des Providers Reference CPU et CUDA.

Les principaux éléments qui avaient motivé le chantier sont donc considérés comme **implémentés** :

- loader Hugging Face externalisé ;
- `config.json` Qwen réel ;
- `tokenizer.json` réel ;
- Safetensors single-file et sharded ;
- stockage F32/F16/BF16 ;
- tied embeddings avec dérivation de `lm_head` ;
- prise en charge des biais Q/K/V Qwen2/Qwen2.5 ;
- Qwen Component configurable ;
- séparation correcte Core / Component / Format / Provider ;
- tests sur un checkpoint public réel `Qwen/Qwen2.5-0.5B-Instruct`.

Cependant, le change a été déclaré terminé et archivé alors que les preuves de clôture ne sont pas encore suffisantes.

Deux anomalies **P0** bloquent le GO de clôture :

1. le workflow **Quality** du SHA final est rouge ;
2. aucune preuve GPU n'a été exécutée sur ce même SHA final.

Un troisième point fonctionnel reste majeur pour l'intégration avec Tachyon-Mesh : le chemin CUDA de production ne supporte encore que le **prefill / premier token**. Le decode multi-token reste non implémenté.

La conclusion est donc :

> **Le Production Model Loading est techniquement quasi validé, mais le change OpenSpec ne doit pas être considéré comme définitivement clôturé tant que la CI du SHA final n'est pas verte et que la preuve GPU correspondante n'a pas été exécutée avec succès.**

Pour Tachyon-Mesh, il reste en outre un bloqueur fonctionnel :

> **Magnetar ne fournit pas encore une génération LLM CUDA multi-token utilisable en production.**

---

## 2. Périmètre de l'audit

L'audit porte sur le résultat final annoncé par l'équipe Magnetar pour le chantier de chargement de modèles Qwen de production.

Les axes vérifiés sont :

1. ingestion d'un bundle Hugging Face réel ;
2. parsing des artefacts de modèle ;
3. configuration dynamique du Qwen Component ;
4. séparation des responsabilités architecturales ;
5. matérialisation des poids ;
6. exécution Reference CPU ;
7. exécution CUDA ;
8. utilisation d'un checkpoint public réel ;
9. absence de retour à un chemin fixture privilégié ;
10. couverture CI ;
11. preuve GPU sur matériel réel ;
12. cohérence du change OpenSpec archivé avec l'état réellement vérifié.

L'audit ne vise pas à valider Magnetar comme runtime généraliste de production pour toutes les architectures, tous les formats ou tous les Providers.

---

## 3. Architecture cible rappelée

La frontière architecturale attendue reste :

```text
Component
    |
    | portable architecture / graph intent
    v
Runtime Core
    |
    | Provider resolution + Resource Affinity
    | Model Instance + Prepared Execution Plan
    | Memory / KV lifecycle contracts
    v
Provider
    |
    | backend-specific implementation
    v
Device
```

Principe fondamental :

> **Core = quoi exécuter, avec quels invariants et sur quelle ressource abstraite.**  
> **Provider = comment l'exécuter sur CUDA / CPU / Metal / QNN / autre backend.**

En particulier :

```text
magnetar-runtime
    X  CUDA implementation
    X  Hugging Face parsing implementation
    X  Safetensors concrete parser dependency
```

Les implémentations spécifiques doivent rester dans les modules externes correspondants.

---

## 4. État global des findings

| ID | Sévérité | Finding | Statut |
|---|---:|---|---|
| ML-P0-1 | P0 | Quality CI du SHA final en échec | ❌ Bloquant |
| ML-P0-2 | P0 | Aucune preuve GPU exécutée sur le SHA final | ❌ Bloquant |
| ML-P1-1 | P1 / P0 Tachyon | CUDA production limité au prefill / premier token | ⚠️ Ouvert |
| ML-P2-1 | P2 | Spec archivée avec un `Purpose` incomplet / TBD | ⚠️ À corriger |
| ML-PASS-1 | PASS | Loader Hugging Face externalisé | ✅ |
| ML-PASS-2 | PASS | Qwen Component configurable | ✅ |
| ML-PASS-3 | PASS | Safetensors single/sharded + F16/BF16/F32 | ✅ |
| ML-PASS-4 | PASS | Tokenizer réel | ✅ |
| ML-PASS-5 | PASS | Checkpoint public réel et pinning | ✅ |
| ML-PASS-6 | PASS | Frontières Core / Provider / Component préservées | ✅ |
| ML-PASS-7 | PASS | Attention bias Q/K/V Qwen2/Qwen2.5 prise en charge | ✅ |

---

# 5. Findings bloquants

## ML-P0-1 — Quality CI du SHA final en échec

### Sévérité

**P0 — Bloquant pour la clôture du change**

### Constat

Le SHA final audité est :

```text
9d57ad4a064dfbfedccbe7360a2216c36b3e19ee
```

Le workflow GitHub Actions associé :

```text
Quality #207
```

est terminé avec :

```text
conclusion = failure
```

La cause identifiée est la suivante :

```text
integration-tests/production-loading/Cargo.lock
```

n'est pas présent alors que la CI exécute :

```bash
cargo test --locked \
  --manifest-path integration-tests/production-loading/Cargo.toml
```

Le job échoue avec une erreur équivalente à :

```text
error: cannot create the lock file
integration-tests/production-loading/Cargo.lock
because --locked was passed
```

### Impact

Le problème n'est pas seulement esthétique.

Comme le job s'arrête à ce point, plusieurs vérifications situées ensuite ne sont pas exécutées, notamment :

- tests de `magnetar-cli` ;
- `cargo check` de `magnetar-cli` ;
- `cargo clippy` de `magnetar-cli` ;
- génération de documentation ;
- source guard empêchant le chemin production de revenir à `fixture_model_manifest` ;
- source guard empêchant l'intégration de type Tachyon de réimplémenter des primitives internes ;
- vérifications de dépendances et de frontières entre le Core et les modules externalisés.

Le fait que les tâches OpenSpec soient cochées ne constitue donc pas une preuve suffisante.

### Correction attendue

Ajouter et commiter le lockfile de l'intégration :

```bash
cd integration-tests/production-loading
cargo generate-lockfile
git add Cargo.lock
```

Puis faire passer le workflow **Quality** entièrement au vert sur un nouveau SHA.

### Critère de clôture

Le finding peut être fermé uniquement si :

```text
HEAD == SHA du workflow Quality
Quality == success
tous les jobs requis == success
```

---

## ML-P0-2 — Absence de preuve GPU sur le SHA final

### Sévérité

**P0 — Bloquant pour la clôture du change**

### Constat

Le workflow :

```text
GPU Runner Smoke Test
```

est maintenant correctement conçu pour valider :

- le Provider CUDA réel ;
- la conformance CUDA ;
- le production-loading CUDA ;
- le checkpoint Qwen public réel ;
- la comparaison Reference CPU / CUDA.

Le workflow télécharge notamment un checkpoint Qwen2.5-0.5B-Instruct à une revision pinée et exécute les tests correspondants sur le runner GPU self-hosted.

Cependant, ce workflow est déclenché uniquement par :

```yaml
on:
  workflow_dispatch:
```

et aucune exécution correspondant au SHA final audité :

```text
9d57ad4a064dfbfedccbe7360a2216c36b3e19ee
```

n'a été trouvée.

### Pourquoi une ancienne preuve GPU ne suffit pas

Le commit final modifie directement plusieurs éléments critiques pour le chemin GPU :

```text
components/qwen
providers/cpu
providers/cuda
loaders/huggingface
magnetar-runtime
```

Il ajoute notamment :

- le support des biais Q/K/V réels ;
- un nouveau broadcast `add` ;
- un kernel CUDA de bias-add ;
- le dispatch Runtime correspondant ;
- les adaptations de shape rules.

La preuve GPU doit donc être exécutée sur **exactement le même SHA** que celui que l'on souhaite déclarer final.

### Correction attendue

Après correction du P0-1 :

1. créer le nouveau commit ;
2. attendre `Quality == success` ;
3. dispatcher `GPU Runner Smoke Test` sur ce SHA ;
4. vérifier que les tests suivants passent réellement.

Le minimum attendu est :

```text
CudaProvider hardware conformance                  PASS
device-resident kernel chaining                   PASS
production loading CUDA E2E                       PASS
real public Qwen checkpoint Reference CPU          PASS
real public Qwen checkpoint CUDA                   PASS
CPU/CUDA greedy prefill result comparison          PASS
```

### Critère de clôture

Le finding est fermé uniquement si :

```text
Quality SHA == GPU Smoke SHA == audited HEAD SHA
Quality == success
GPU Runner Smoke Test == success
```

---

# 6. Finding fonctionnel majeur

## ML-P1-1 — Le CUDA production ne supporte pas encore le decode multi-token

### Sévérité Magnetar

**P1 — important, non bloquant pour le seul change "Production Model Loading"**

### Sévérité Tachyon-Mesh

**P0 — bloque le remplacement complet de Candle par Magnetar pour l'inférence CUDA**

### Constat

Le test du vrai checkpoint CUDA utilise actuellement une requête équivalente à :

```text
max_tokens = 1
```

Le chemin validé est donc essentiellement :

```text
prompt
  |
  v
prefill CUDA
  |
  v
logits
  |
  v
sample token #1
```

En revanche :

```text
token #1
  |
  v
decode CUDA avec historique KV
  |
  v
token #2
```

n'est pas encore supporté dans le profil de production.

La limitation est explicitement documentée dans le repository.

Le problème vient du fait que la logique actuelle de decode historique attend une représentation host-readable de l'historique KV, alors que le Provider CUDA maintient volontairement ses tensors dans une représentation Device-resident.

### Pourquoi ce point ne doit pas être "corrigé" par des copies systématiques

La mauvaise solution serait :

```text
Device KV
   |
   | D2H
   v
Host concatenation
   |
   | H2D
   v
Device KV
```

à chaque token.

Cela casserait le principe de hot path CUDA Device-resident précédemment validé.

### Architecture cible recommandée

Le KV cache doit rester physiquement Device-resident :

```text
CUDA Device
   |
   +-- KV cache allocation
   |
   +-- append K/V for new token
   |
   +-- attention reads logical KV window
   |
   +-- next-token logits
```

Le Runtime conserve :

- les identités ;
- les transactions ;
- le lifecycle ;
- la Resource Affinity ;
- les invariants du cache.

Le Provider CUDA conserve :

- l'allocation physique ;
- les buffers ;
- les kernels ;
- la mise à jour / append ;
- l'accès au KV historique sur Device.

### Change recommandé

Créer un nouveau change OpenSpec séparé, par exemple :

```text
make-cuda-kv-cache-fully-device-resident
```

ou :

```text
implement-device-resident-multi-step-cuda-decode
```

### Preuve E2E attendue

```text
real Qwen2.5 checkpoint
        |
        v
real tokenizer
        |
        v
real Qwen Component
        |
        v
real ModelInstance
        |
        v
CudaProvider
        |
        v
prefill
        |
        v
decode token 1
        |
        v
decode token 2
        |
        v
decode token 3
        |
        v
...
        |
        v
8 / 16+ generated tokens
```

Assertions :

```text
Provider == CUDA
Device == real GPU
no Reference CPU fallback
KV remains Device-resident
no historical KV D2H/H2D loop
bounded allocation growth
correct resource teardown
numerical/token result compatible with Reference CPU
```

---

# 7. Finding documentation / gouvernance

## ML-P2-1 — Spec canonique avec Purpose incomplet

### Sévérité

**P2 — documentation / qualité OpenSpec**

### Constat

La spec canonique issue de l'archivage du change contient encore un `Purpose` généré ou incomplet de type :

```text
Purpose
TBD - created by archiving change ...
```

### Impact

Aucun impact Runtime direct.

En revanche, cela réduit la qualité de la documentation canonique et laisse un change de cette importance avec une spec finale moins précise que le contenu réel livré.

### Correction recommandée

Rédiger le Purpose final en précisant que la capability couvre :

- ingestion d'un bundle Qwen Hugging Face local autorisé ;
- conversion vers les contrats génériques de Magnetar ;
- tokenizer réel ;
- Safetensors ;
- configuration d'architecture ;
- production `ModelInstance` / inference path ;
- sans coupler le Core à Hugging Face.

---

# 8. Éléments validés

## ML-PASS-1 — Loader Hugging Face correctement externalisé

La logique Hugging Face est portée par :

```text
loaders/huggingface
```

et ne devient pas une dépendance concrète du Core.

Cette séparation est conforme à l'architecture Magnetar.

### Verdict

**PASS**

---

## ML-PASS-2 — Qwen Component réellement configurable

Le Qwen Component n'est plus limité aux constantes de la fixture initiale.

La configuration de production inclut notamment :

```text
hidden_size
intermediate_size
layer_count
attention_heads
kv_heads
head_dimension
vocabulary
rope parameters
RMSNorm epsilon
tie_word_embeddings
attention_bias
```

Le Component reste indépendant du Provider et du Device.

### Verdict

**PASS**

---

## ML-PASS-3 — Safetensors production

Le nouveau chemin couvre :

- fichier Safetensors unique ;
- bundle shardé avec index Hugging Face ;
- validation de layout ;
- validation des ranges ;
- validation des digests ;
- F32 ;
- F16 ;
- BF16.

Les poids F16/BF16 sont actuellement convertis vers F32 pour le compute, ce qui est acceptable pour ce premier profil de production.

### Verdict

**PASS pour le profil déclaré**

---

## ML-PASS-4 — Tokenizer réel

Le chemin production utilise un vrai tokenizer issu de :

```text
tokenizer.json
```

via le module loader externe.

Le Core continue de dépendre uniquement de son contrat générique de tokenizer.

### Verdict

**PASS**

---

## ML-PASS-5 — Checkpoint public réel

Le repository contient maintenant des tests utilisant :

```text
Qwen/Qwen2.5-0.5B-Instruct
```

avec :

- revision pinée ;
- vérification de contenu / digest ;
- vraie configuration ;
- vrai tokenizer ;
- vrais poids ;
- vraie exécution Reference CPU ;
- chemin CUDA prévu dans le GPU smoke test.

L'utilisation de ce checkpoint a effectivement permis de découvrir des écarts absents des fixtures, notamment les biais Q/K/V.

### Verdict

**PASS côté implémentation**

**La preuve CUDA reste néanmoins à rejouer sur le SHA final.**

---

## ML-PASS-6 — Frontières Core / Provider / Component

Aucune régression architecturale majeure n'a été identifiée dans le scope audité.

Le Core reste l'autorité sur :

```text
Provider abstraction
Device abstraction
Resource Affinity
Model Instance
Prepared Execution Plan
Tensor Resource
Memory lifecycle
KV lifecycle contracts
generic execution
```

Les détails CUDA restent dans le Provider CUDA.

Les détails Hugging Face restent dans le loader.

La topologie Qwen portable reste dans le Component.

### Verdict

**PASS**

---

## ML-PASS-7 — Attention bias Qwen2/Qwen2.5

Le test sur modèle réel a révélé que les projections Q/K/V de Qwen2/Qwen2.5 utilisent un bias.

La correction a été apportée de manière cohérente :

```text
Hugging Face loader
        |
        +-- détecte les bias tensors
        |
Qwen Component
        |
        +-- génère le broadcast add
        |
Runtime shape validation
        |
        +-- RowBroadcastAdd
        |
Reference CPU Provider
        |
        +-- broadcast add
        |
CUDA Provider
        |
        +-- bias_add_kernel
```

Cette correction est saine architecturalement.

### Verdict

**PASS**

---

# 9. Éléments qui ne doivent pas être rouverts dans ce change

Les sujets suivants sont considérés suffisamment implémentés pour ne pas être remis dans le scope du change actuel :

- loader Hugging Face externe ;
- parsing `config.json` Qwen ;
- tokenizer réel ;
- single-file Safetensors ;
- sharded Safetensors ;
- support storage F16/BF16 ;
- tied embeddings ;
- dérivation de `lm_head` ;
- Q/K/V bias ;
- Qwen Component configurable ;
- production model loading public API ;
- séparation Core / loader / Component / Provider ;
- utilisation d'un vrai checkpoint Qwen public.

Les limitations suivantes sont acceptables si elles restent explicitement documentées comme hors scope du profil v0.1 :

- GGUF pour ce chemin de production Qwen ;
- quantization ;
- LoRA ;
- autres architectures ;
- remote hub download ;
- native CUDA FP16/BF16 kernels ;
- multi-device inference.

---

# 10. Plan de remédiation demandé avant réaudit

## Étape 1 — Corriger la CI

Ajouter :

```text
integration-tests/production-loading/Cargo.lock
```

et produire un nouveau commit.

## Étape 2 — Obtenir un Quality entièrement vert

Sur le nouveau SHA :

```text
Quality
  |
  +-- root checks
  +-- component integration
  +-- format integration
  +-- provider integration
  +-- submodule integration
  +-- magnetar-cli
  +-- source guards
  +-- dependency guards
```

doit être entièrement vert.

## Étape 3 — Rejouer le GPU Runner Smoke Test

Dispatcher manuellement le workflow GPU sur **ce même SHA exact**.

Le run doit prouver :

```text
real CUDA hardware visible
CudaProvider conformance executed
device-resident hot path executed
production loading CUDA E2E executed
real public Qwen checkpoint executed
Reference CPU and CUDA comparison executed
```

## Étape 4 — Corriger la documentation OpenSpec

Corriger le `Purpose` de la spec canonique concernée.

## Étape 5 — Documenter les preuves

Conserver dans l'audit / change closeout :

```text
Final HEAD SHA
Quality workflow run ID
Quality conclusion
GPU Runner Smoke Test run ID
GPU runner identity
CUDA GPU model
CUDA/driver version
real checkpoint revision
checkpoint digest
CPU test result
CUDA test result
```

---

# 11. Critères de GO pour le change Production Qwen Model Loading

Le change pourra être déclaré **GO** lorsque les conditions suivantes seront toutes réunies :

```text
[ ] HEAD final connu et immuable pour l'audit
[ ] Quality passe entièrement sur ce HEAD
[ ] GPU Runner Smoke Test passe sur ce même HEAD
[ ] vrai checkpoint Qwen public chargé
[ ] vrai tokenizer utilisé
[ ] vrai Qwen Component utilisé
[ ] vrai ModelInstance utilisé
[ ] Reference CPU path passe
[ ] real CUDA Provider path passe
[ ] aucune fallback CPU silencieuse dans le test CUDA
[ ] source guards passent
[ ] OpenSpec canonical docs corrigées
```

Verdict attendu après satisfaction de ces critères :

> **GO — Production Qwen Model Loading closable**

---

# 12. Critères de GO séparés pour le cutover Tachyon-Mesh

Le GO du Production Model Loading ne suffit pas à autoriser le remplacement complet de Candle par Magnetar dans Tachyon-Mesh.

Pour Tachyon, il faut en plus :

```text
[ ] real multi-token CUDA decode
[ ] Device-resident KV cache
[ ] no historical-KV host round-trip
[ ] streaming generation path
[ ] real Tachyon request -> Magnetar Runtime
[ ] real ModelInstance
[ ] real PreparedExecutionPlan
[ ] real CUDA Provider / Device
[ ] 8/16+ generated tokens
[ ] no Reference CPU fallback
[ ] result parity / compatibility against Reference CPU
[ ] bounded resource growth
[ ] clean teardown
```

Le chemin E2E attendu est :

```text
Tachyon Node A
      |
      v
mesh routing
      |
      v
Tachyon Node B
      |
      v
Magnetar Runtime
      |
      v
real Qwen ModelInstance
      |
      v
compiled Qwen Component
      |
      v
Prepared Execution Plan
      |
      v
CudaProvider
      |
      v
real CUDA Device
      |
      v
prefill + multi-step decode
      |
      v
generated token stream
      |
      v
Tachyon Node A
```

Tant que ce chemin multi-token n'existe pas, le verdict Tachyon reste :

> **NO-GO — full local CUDA inference cutover**

---

# 13. Verdict final

## Production Qwen Model Loading

**CHANGES REQUIRED — NO-GO de clôture**

L'implémentation technique est suffisamment avancée pour considérer que le cœur du chantier a été livré.

Les deux bloqueurs restants concernent principalement la qualité de clôture et les preuves :

1. CI Quality rouge ;
2. absence de run GPU sur le SHA final.

Ces deux P0 sont relativement simples à fermer.

Après correction et preuve au SHA exact, le change pourra vraisemblablement passer en :

> **GO — archive justifiée**

## Tachyon-Mesh

Pour le remplacement complet de Candle par Magnetar :

> **NO-GO**

Le principal bloqueur restant n'est désormais plus le Model Loading.

Il est :

> **le decode CUDA multi-step avec KV cache Device-resident.**

Le prochain chantier recommandé est donc :

```text
implement-device-resident-multi-step-cuda-decode
```

et non une nouvelle refonte du Production Model Loading.

---

# 14. Synthèse à transmettre à l'équipe

Le chantier `implement-production-qwen-model-loading` a produit une implémentation techniquement crédible et largement conforme à l'architecture Magnetar.

Le gros travail de fond est validé.

Avant clôture définitive, merci de :

1. ajouter le lockfile manquant de `integration-tests/production-loading` ;
2. obtenir une Quality CI complètement verte ;
3. rejouer le GPU Runner Smoke Test sur exactement le même SHA ;
4. conserver les IDs et preuves de ces runs ;
5. corriger le `Purpose` OpenSpec restant incomplet.

Ensuite, le Production Model Loading pourra être considéré clos.

Le prochain sujet fonctionnel prioritaire pour permettre le cutover Tachyon-Mesh est le **multi-step CUDA decode avec KV cache Device-resident**.
