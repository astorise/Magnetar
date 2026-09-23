# Audit de clôture Magnetar — 23 septembre 2026

**Repository :** `astorise/Magnetar`  
**SHA `main` audité :** `9db481de9dd43efcd54d06b62f0b90d6cc61aca3`  
**SHA vendored par Tachyon :** `9db481de9dd43efcd54d06b62f0b90d6cc61aca3`  
**Verdict :** 🟢 **GO sur le périmètre Magnetar audité**

## Invariant

Magnetar reçoit de Tachyon un Component WASM explicite. Magnetar reste propriétaire du Model Artifact, format, tokenizer, architecture, ModelInstance, Providers, Devices, Kernels, prefill/decode, KV cache, quantization, sampling, génération et streaming.

La façade exposée à Tachyon doit rester architecture-neutral.

## État de l’intégration

Le SHA audité est exactement celui consommé par Tachyon :

```text
Magnetar main
9db481de9dd43efcd54d06b62f0b90d6cc61aca3

Tachyon vendor/Magnetar
9db481de9dd43efcd54d06b62f0b90d6cc61aca3
```

✅ Plus aucune divergence de pin.

## Correctifs du précédent audit

Les deux derniers findings Magnetar étaient :

```text
MAG-01 — qwen_component_runtime_limits() dans le registre générique
MAG-02 — guard générique ne couvrant pas register_inference_component_artifact()
```

Ils ont été corrigés par :

```text
19d5893469ea93d918c65da5496f762363d13d25
Generalize the Component registry's runtime-limits helper name (MAG-01/MAG-02) (#95)
```

PR : https://github.com/astorise/Magnetar/pull/95

## MAG-01 — ✅ Clos

L’ancien helper :

```rust
qwen_component_runtime_limits()
```

est devenu :

```rust
inference_component_runtime_limits()
```

Le registre générique appelle désormais :

```rust
manager.set_resource_limits(inference_component_runtime_limits());
```

Le singleton Qwen historique utilise le même helper générique.

La documentation précise que :

- les limites ne dépendent d’aucune famille ;
- le Component Llama utilise les mêmes valeurs ;
- toute future différence doit être exprimée par une politique générique ou le manifeste ;
- aucune branche famille-spécifique ne doit être créée.

✅ Le registre générique n’est plus Qwen-shaped sur ce point.

## MAG-02 — ✅ Clos

`tools/check_generic_facade_family_isolation.py` couvre maintenant :

```text
ProductionLoadedModel
production_model_fixture
build_first_native_graphs_from_named_component
register_inference_component_artifact
LoadedInferenceComponent::load
```

Le guard inspecte les identifiants de familles comme :

```text
qwen
llama
gemma
mistral
mixtral
phi
```

dans les corps génériques, avec allowlist explicite pour les exceptions justifiées.

✅ L’ancien angle mort du registre est fermé.

## ProductionLoadedModel — ✅ générique

Le chemin générique n’est plus structuré autour de :

```text
QwenConfig
production_qwen_component_identity
qwen_component_descriptor
qwen_validate_model_artifact
execute_qwen_graph
QwenLayerKvMap
```

Les éléments Qwen-spécifiques qui subsistent sont confinés aux chemins explicitement Qwen : singleton historique, fixtures, tests et Component Qwen.

✅ Conforme.

## Vraie seconde architecture — ✅

Le test :

```rust
loaded_inference_component_load_runs_a_real_second_architecture_end_to_end()
```

utilise un vrai bundle Llama distinct :

```rust
write_tiny_llama_bundle(...)
```

avec config, dimensions, tokenizer, tensors, Component WASM et digest distincts.

Chemin :

```text
LoadedInferenceComponent::load
→ ProductionLoadedModel
→ Component graph
→ Provider
→ generation
```

✅ L’ancien faux Llama sur Artifact Qwen reste fermé.

## Compatibilité architecture — ✅

Le manifeste Component peut déclarer :

```yaml
compatibility:
  architecture_families:
    - llama
```

ou :

```yaml
compatibility:
  architecture_families:
    - qwen2
```

Cas négatif obligatoire :

```text
Llama Artifact
+ Qwen Component [qwen2]
→ ArchitectureUnsupported
```

Cas positif :

```text
Llama Artifact
+ Llama Component [llama]
→ succès
```

✅ #83 reste correctement fermé.

## Artifact Format — ✅

Magnetar exige désormais :

```text
magnetar-artifact-format.yaml
```

par exemple :

```yaml
artifact_format: huggingface
```

ou :

```yaml
artifact_format: gguf
```

Le runtime utilise :

```rust
read_declared_artifact_format(...)
```

et ne sélectionne plus l’ingestor par heuristique filesystem dans le chemin normal.

Le Component peut également déclarer :

```yaml
compatibility:
  artifact_formats:
    - huggingface
```

Une incompatibilité produit :

```text
ArtifactFormatUnsupported
```

✅ Conforme.

## Trust — ✅

Le trust Component et le trust Model Artifact restent séparés :

```text
Component WASM
    └── ComponentTrustStore

Model Artifact
    └── ModelTrustStore
```

`register_inference_component_artifact()` :

1. calcule le digest réel ;
2. parse le manifeste ;
3. vérifie le digest déclaré ;
4. consulte le cache de compilation ;
5. réévalue le trust du caller même sur cache hit.

Le cache ne confère jamais de trust implicite.

✅ Conforme.

## Autorité du Component WASM — ✅

Le Component fourni par Tachyon est enregistré par :

```rust
register_inference_component_artifact(...)
```

puis utilisé par :

```rust
build_first_native_graphs_from_named_component(...)
```

Le chemin générique ne sélectionne pas un Component Qwen par défaut.

✅ Conforme.

## Multiplicité — ✅

Plusieurs Components distincts peuvent coexister dans le registry indexé par digest.

La preuve Qwen/Llama est également exercée côté Tachyon avec exactement ce SHA Magnetar.

✅ Conforme.

## CI

SHA :

```text
9db481de9dd43efcd54d06b62f0b90d6cc61aca3
```

Workflow :

```text
Quality — 35885337508
completed / success
```

Run : https://github.com/astorise/Magnetar/actions/runs/35885337508

Les gates historiquement problématiques sont verts :

```text
wasm32 component engine ✅
docs                    ✅
coverage                ✅
model-family isolation  ✅
component integration   ✅
format integration      ✅
provider integration    ✅
e2e conformance         ✅
WIT                     ✅
OpenSpec                ✅
```

## Findings actuels

### Aucun P0 identifié

Je ne retrouve plus les anciens blockers :

```text
Component trust / Artifact trust confondus
cache-hit trust bypass
WASM Component bypass
ProductionLoadedModel Qwen-shaped
API générique QwenConfig
faux Llama sur Artifact Qwen
architecture compatibility absente
artifact format implicite
wasm32 cassé
docs rouges
coverage rouge
registre générique appelant qwen_component_runtime_limits
guard générique aveugle au registre
```

Tous sont fermés sur le SHA audité.

## Critères de GO Magnetar

| Critère | État |
|---|:---:|
| Component explicite reçu | ✅ |
| Component trust séparé | ✅ |
| Artifact trust séparé | ✅ |
| Cache-hit réévalue le trust | ✅ |
| Component réel autorité du graphe | ✅ |
| Plusieurs Components coexistants | ✅ |
| API de graph générique | ✅ |
| `ProductionLoadedModel` architecture-neutral | ✅ |
| Vrai Artifact Llama | ✅ |
| Vrai Component Llama | ✅ |
| Incompatibilité Qwen/Llama rejetée | ✅ |
| Artifact Format explicite | ✅ |
| Format non inféré par filesystem | ✅ |
| Registry générique sans helper Qwen | ✅ |
| Guard couvre le registry générique | ✅ |
| `wasm32` vert | ✅ |
| docs vertes | ✅ |
| coverage vert | ✅ |
| Quality du SHA intégré verte | ✅ |
| SHA Magnetar = pin Tachyon | ✅ |

## Verdict

🟢 **GO Magnetar**

Sur le périmètre audité, Magnetar satisfait désormais la frontière attendue.

```text
Tachyon
    │ Component WASM explicite
    │ opaque payload
    │ trust
    │ placement générique
    ▼
Magnetar
    ├── Component manifest
    ├── architecture compatibility
    ├── Artifact format
    ├── Model Artifact
    ├── ModelInstance
    ├── Provider / Device
    ├── tokenizer
    ├── graph execution
    └── generation
```

Aucun nouveau blocker Magnetar n’a été identifié dans cette passe. Les problèmes restants de l’intégration sont côté Tachyon : exposition de détails Provider et vocabulaire interne `generate/prompt`.
