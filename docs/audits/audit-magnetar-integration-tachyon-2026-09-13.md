# Audit de clôture Magnetar – Intégration Tachyon

**Date :** 13 septembre 2026
**Repository :** `astorise/Magnetar`
**Commit consommé par Tachyon :** `ee7ef9e9e414f3b59ea91b11a2f519bfb5085152`
**Branche :** `codex/tachyon-component-boundary`
**Magnetar `main` observé lors de l'audit :** `a1f0771c56f7dc1a8192a6334f1dcb04ed72be81`
**Verdict :** 🔴 **NO-GO pour clôture de l'intégration Tachyon – Magnetar**

---

## 1. Invariant d'architecture

Magnetar doit être le runtime capable de recevoir un **Component WASM d'inférence explicite** fourni par Tachyon, puis d'exécuter ce Component sans faire d'hypothèse sur :

- la famille de modèle ;
- le format du modèle ;
- le tokenizer ;
- le chat template ;
- le type concret de `ModelInstance` ;
- le Provider concret ;
- les kernels ;
- le format Hugging Face, GGUF, Safetensors, etc.

Architecture cible :

```text
Tachyon
   │
   │ Component WASM + manifest
   │ trust policy + placement
   │ payload opaque
   ▼
Magnetar
   │
   │ generic Component Engine
   ▼
Inference Component
   │
   ├── modèle
   ├── architecture
   ├── tokenizer
   ├── Provider
   ├── kernels
   └── génération
```

Critère clé :

> Le même Magnetar doit pouvoir exécuter deux Components d'inférence distincts sans branchement par famille de modèle dans l'adapter générique.

---

## 2. Points désormais corrects

### ✅ Le Component d'inférence est explicite

`LoadedInferenceComponent::load(...)` reçoit désormais explicitement un `InferenceComponentArtifact`.

Le précédent mécanisme de Component Qwen embarqué par défaut a été supprimé.

### ✅ Plus de Component implicite par défaut

La logique :

```text
DEFAULT_COMPONENT_BYTES
DEFAULT_COMPONENT_MANIFEST_BYTES
register_default_component
```

a été supprimée du chemin générique.

### ✅ La frontière publique est plus propre

La façade expose maintenant des types génériques comme :

```text
InferenceComponentArtifact
InferenceComponentSource
InferenceComponentPlacement
ArtifactTrustPolicy
LoadedInferenceComponent
```

C'est un progrès net par rapport à l'ancienne intégration directement Qwen/Hugging Face.

---

# 3. MAG-01 – `LoadedInferenceComponent` reste Qwen/HuggingFace spécifique

**Sévérité : P0 / Critical**
**Statut : 🔴 Merge blocker**

Malgré son nom générique, l'implémentation importe et utilise directement :

```text
HuggingFaceIngestor
HuggingFaceChatTemplateFormatter
HuggingFaceTokenizer
ReferenceCpuProvider
ProductionQwenLoadedModel
```

Le state contient :

```rust
loaded_model: Mutex<magnetar_runtime::ProductionQwenLoadedModel>
```

Le chargement exécute systématiquement :

```text
HuggingFace ingestion
tokenizer.json
tokenizer_config.json
production_qwen_fixture(...)
ProductionQwenLoadedModel::load(...)
```

et l'enregistrement du Component passe encore par :

```rust
register_qwen_component_artifact(...)
```

## Impact

La façade est générique en nom, mais pas en comportement.

Aujourd'hui elle correspond davantage à :

```text
LoadedQwenHuggingFaceInferenceComponent
```

qu'à un vrai :

```text
LoadedInferenceComponent
```

## Correction requise

Le chemin générique ne doit contenir aucune référence à :

```text
Qwen
HuggingFace
tokenizer.json
ProductionQwenLoadedModel
production_qwen_fixture
```

Ces détails doivent être résolus derrière le Component ou une couche Magnetar spécifique au modèle, jamais dans l'adapter générique.

---

# 4. MAG-02 – Le Component WASM fourni n'est pas l'autorité réelle d'exécution

**Sévérité : P0 / Critical**
**Statut : 🔴 Merge blocker**

Le Component WASM fourni par Tachyon est bien reçu et enregistré.

Mais `invoke_payload()` appelle ensuite directement :

```text
ProductionQwenLoadedModel::generate(...)
```

et le streaming :

```text
ProductionQwenLoadedModel::generate_streaming(...)
```

Le flux réel reste donc :

```text
Component WASM
     │
     ├── enregistré / validé
              │
              ▼
       ProductionQwenLoadedModel
              │
              ▼
         génération native
```

au lieu de :

```text
Component WASM
     │
     ▼
Component Engine
     │
     ▼
invoke(Component)
     │
     ▼
résultat
```

## Correction requise

Le Component fourni doit devenir l'autorité d'exécution.

La façade générique doit effectuer quelque chose conceptuellement équivalent à :

```text
engine.load(component)
engine.instantiate(component)
engine.invoke(payload)
```

Les appels directs à un modèle natif concret ne doivent plus exister dans ce chemin.

---

# 5. MAG-03 – Le trust « Artifact/Component » est appliqué au modèle

**Sévérité : P0 / Security**
**Statut : 🔴 Merge blocker**

La façade expose :

```rust
ArtifactTrustPolicy
```

mais ce type encapsule actuellement :

```rust
ModelTrustStore
```

Puis la décision de trust est appliquée à :

```text
ingested.manifest
```

c'est-à-dire au manifest du modèle produit par l'ingestion Hugging Face.

Or Magnetar possède déjà des primitives adaptées :

```text
ComponentTrustStore
ComponentDigest
ComponentManifest
ComponentTrustDecision
```

## Risque

La configuration côté Tachyon peut donner l'impression qu'elle autorise un **Component exécutable**, alors que la décision porte réellement sur un digest de Model Artifact.

Ces deux décisions de confiance doivent être indépendantes.

## Architecture attendue

```text
Tachyon trust
    └── Component WASM / executable artifact

Magnetar trust
    └── model / weights / tokenizer / dependent artifacts
```

## Correction requise

Le chemin d'intégration doit utiliser :

```text
ComponentDigest
ComponentManifest
ComponentTrustStore
```

pour le WASM exécutable.

Le `ModelTrustStore` peut continuer à exister derrière Magnetar pour les artefacts de modèle.

---

# 6. MAG-04 – Branche d'intégration divergente de `main`

**Sévérité : P1 / High**

Tachyon consomme actuellement :

```text
codex/tachyon-component-boundary
ee7ef9e9...
```

alors que `main` Magnetar observé pendant l'audit est :

```text
a1f0771c...
```

Les deux historiques ont divergé.

## Impact

Les corrections d'intégration Tachyon ne sont pas réconciliées avec la ligne principale Magnetar qui continue à évoluer sur :

- first-native runtime ;
- loaders ;
- GGUF ;
- kernels ;
- CUDA ;
- model loading ;
- Component engine.

## Correction requise

- intégrer les changements `tachyon-component-boundary` dans `main` ;
- ou rebased/merge proprement la branche sur `main` ;
- faire consommer à Tachyon un commit issu de la lignée principale qualifiée.

---

# 7. MAG-05 – CI du commit consommé

**Sévérité : P1 / High**

`main` Magnetar dispose d'une Quality CI complète.

Le commit `ee7ef9e9...` consommé par Tachyon n'a pas de workflow GitHub Actions propre attaché.

Il est bien compilé/testé indirectement dans la CI Tachyon, mais cela ne remplace pas la Quality matrix Magnetar complète.

## Correction requise

Le commit exact consommé par Tachyon doit passer :

```text
cargo check
clippy -D warnings
cargo test
cargo deny
MSRV
wasm32 Component runtime
provider conformance
OpenSpec validation
integration tests concernés
```

---

# 8. MAG-06 – Sérialisation des générations

**Sévérité : P2 / Medium**

`LoadedInferenceComponent` garde encore :

```rust
Mutex<ProductionQwenLoadedModel>
```

et conserve le verrou autour de la génération.

Cela semble sérialiser les requêtes sur une même instance résidente.

Ce point n'est pas un problème Tachyon : il appartient totalement à Magnetar.

## Action recommandée

Valider explicitement le modèle de concurrence attendu :

- une génération active par instance ;
- scheduler interne ;
- batching ;
- sessions concurrentes ;
- duplication contrôlée d'instances.

---

# 9. MAG-07 – Test architectural à ajouter

**Sévérité : P1 / High**

Ajouter un test qui prouve :

```text
Component A.wasm ──┐
                    ├──► même Magnetar
Component B.wasm ──┘
```

sans modification de code et sans branchement sur la famille.

Les deux Components n'ont pas besoin d'être deux gros LLM de production.

Un Component synthétique est suffisant si les deux artefacts produisent des comportements distincts.

Le test doit échouer si l'adapter contient un chemin codé en dur Qwen.

---

# 10. Scope correctif proposé

```markdown
## Final Magnetar Component Runtime Closure

- [ ] Make `LoadedInferenceComponent` truly model-agnostic.
- [ ] Remove direct HuggingFace and Qwen assumptions from the generic adapter.
- [ ] Remove `ProductionQwenLoadedModel` from the generic Component path.
- [ ] Execute the explicit WASM Component through the Component Engine.
- [ ] Replace `register_qwen_component_artifact` with generic Component registration/loading.
- [ ] Apply executable trust with `ComponentTrustStore` / `ComponentDigest`.
- [ ] Keep model artifact trust as a separate Magnetar-owned concern.
- [ ] Reconcile `codex/tachyon-component-boundary` with Magnetar `main`.
- [ ] Run the complete Magnetar Quality matrix on the exact commit consumed by Tachyon.
- [ ] Add a two-distinct-Components test using the same Magnetar binary.
- [ ] Document the intended concurrent generation model and remove unnecessary global serialization.
```

---

# 11. Critères de GO Magnetar

| Critère | État |
|---|:---:|
| Component explicite obligatoire | ✅ |
| Aucun Component par défaut compilé | ✅ |
| API publique Component-centric | ✅ |
| Adapter générique sans Qwen | ❌ |
| Adapter générique sans HuggingFace | ❌ |
| Component WASM réellement invoqué | ❌ |
| Component trust réellement appliqué au WASM | ❌ |
| Model trust séparé du Component trust | ❌ |
| Branche d'intégration réconciliée avec `main` | ❌ |
| CI Magnetar complète sur le SHA consommé | ❌ |
| Deux Components distincts prouvés | ❌ |

---

# 12. Verdict

## 🔴 NO-GO Magnetar

Le contrat public a été nettement amélioré et le Component est maintenant fourni explicitement.

Mais le chemin d'exécution réel reste encore :

```text
generic façade
      │
      ▼
HuggingFace
      │
      ▼
Qwen
      │
      ▼
ProductionQwenLoadedModel
```

La clôture architecture sera atteinte lorsque le chemin réel sera :

```text
generic façade
      │
      ▼
Component Engine
      │
      ▼
arbitrary inference Component
```

sans connaissance Qwen/Hugging Face dans l'adapter générique.

---

# 13. Suivi de clôture (post-audit)

Traité côté Magnetar en trois chantiers, dans cet ordre :

1. **MAG-04** : `codex/tachyon-component-boundary` réconciliée avec `main` (merge commit `3f101ae`).
2. **`wire-generic-inference-component-runtime`** (Phase A) : registre de Components générique et digest-keyed dans `magnetar-runtime` (`register_inference_component_artifact`/`build_first_native_graphs_from_named_component`), vérifié correct contre le chemin singleton pré-existant. Le registre n'a pas encore de consommateur réel à ce stade.
3. **`wire-inference-component-to-generic-registry`** : `inference-components::LoadedInferenceComponent` câblé sur ce registre -- `ArtifactTrustPolicy` sépare réellement le trust du Component (`ComponentTrustStore`) de celui du modèle (`ModelTrustStore`, MAG-03), et `ProductionQwenLoadedModel::load_with_component` fait du Component enregistré l'autorité d'exécution réelle pour la production de graphe (plan et dispatch), pas seulement à l'enregistrement (MAG-02). Vérifié par un test de bout en bout : génération de tokens identiques entre le chemin singleton et le chemin Component-enregistré, pour les mêmes octets réels.

4. **`enable-pluggable-model-ingestion-in-inference-components`** : `LoadedInferenceComponent::load` sélectionne désormais entre `GgufIngestor` et `HuggingFaceIngestor` (tous deux implémentant déjà le même contrat `ProductionModelArtifactIngestor` format-neutre) en inspectant le bundle reçu, au lieu de câbler HuggingFace en dur comme seul chemin atteignable -- clôt MAG-01. Le rendu du chat template n'a nécessité aucun code spécifique au format : `loaders/gguf` renvoie déjà le texte brut du template pour le même formateur générique que le chemin HuggingFace utilisait déjà.

5. **`document-inference-component-concurrency-model`** : investigation d'abord (plutôt que choisir une politique arbitraire) -- `ProductionQwenLoadedModel::generate`/`generate_streaming` prennent déjà `&mut self`, donc "une génération active par instance" était déjà garanti par le compilateur, pas une question de conception encore ouverte. Ce chantier le documente explicitement sur `LoadedInferenceComponent`, sans changer une seule ligne de comportement -- clôt MAG-06.

**Reste ouvert** : MAG-07 (aucun test avec deux Components réels et distincts -- un seul fixture réel `model-component-graph-producer` existe dans le repo à ce jour), et un test d'intégration dédié dans `inference-components` pour `LoadedInferenceComponent::load` lui-même, pour l'un ou l'autre format (le chemin magnetar-runtime équivalent est testé, mais l'orchestration propre à cette crate -- ingestion, trust, construction de fixture -- ne l'est pas encore).
