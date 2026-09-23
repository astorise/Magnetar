# Audit Magnetar — 23 septembre 2026

**Repository :** `astorise/Magnetar`  
**SHA `main` audité :** `7e1be5b23e1931579967701ba789f60d585bec09`  
**SHA actuellement vendored par Tachyon :** `7b4861613c8da5b1ebf8816bd4e3d66065c9e440`  
**Verdict :** 🟢 **GO fonctionnel / 🟠 cleanup architectural mineur**

## Invariant

Magnetar reçoit de Tachyon un Component WASM explicite et reste propriétaire de toute la sémantique d’inférence : Model Artifact, format, tokenizer, architecture, ModelInstance, Providers, Devices, Kernels, prefill/decode, KV cache, quantization, sampling, génération et streaming.

La façade utilisée par Tachyon doit cependant rester architecture-neutral.

## Points désormais validés

- Component WASM explicitement enregistré.
- Component réel autorité de production des graphes.
- Registry multi-Components indexé par digest.
- Trust Component séparé du trust Model Artifact.
- Cache-hit réévalue l’autorisation.
- API générique de graph en `ModelArchitectureConfig`.
- `ProductionLoadedModel` dé-Qwenifié.
- Vrai Artifact Llama indépendant.
- Vrai Component Llama indépendant.
- Test Llama via `LoadedInferenceComponent::load`.
- Compatibilité Component ↔ architecture family dans le manifeste.
- `Llama Artifact + Qwen Component` rejeté.
- `Llama Artifact + Llama Component` accepté.
- Artifact Format désormais explicite.
- Détection filesystem retirée du chemin normal.
- `wasm32`, docs, coverage et Quality verts.

## Architecture family compatibility

Le manifeste peut déclarer :

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

Le Runtime transporte cette valeur dans `ComponentManifest.supported_architecture_families` et la vérifie sur le vrai chemin de chargement.

Cas négatif :

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

L’issue #83 est correctement fermée.

## Seconde architecture réelle

`loaded_inference_component_load_runs_a_real_second_architecture_end_to_end()` utilise désormais `write_tiny_llama_bundle(...)` avec config, dimensions, tokenizer et tensors distincts.

Le chemin testé est :

```text
LoadedInferenceComponent::load
→ ProductionLoadedModel
→ Component graph
→ Provider
→ génération
```

L’ancien « faux Llama sur bundle Qwen » est clos.

## Trust

`register_inference_component_artifact()` calcule le digest réel, parse le manifeste, vérifie le digest, puis réévalue `trust.evaluate(&manifest, &digest)` même sur cache hit.

Le cache évite uniquement une recompilation. Il ne confère jamais de trust implicite.

## Artifact Format

Magnetar `main` exige désormais un sidecar explicite :

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

Le Runtime utilise `read_declared_artifact_format(...)`.

Le Component peut aussi déclarer :

```yaml
compatibility:
  artifact_formats:
    - huggingface
```

et une incompatibilité produit `ArtifactFormatUnsupported`.

Le chemin normal n’infère plus le format depuis le filesystem.

## Findings

### MAG-01 — 🟠 P1 — Le registre générique utilise encore `qwen_component_runtime_limits()`

Dans :

```rust
register_inference_component_artifact(...)
```

le chemin générique appelle encore :

```rust
manager.set_resource_limits(qwen_component_runtime_limits());
```

Les valeurs sont génériques en forme et le vrai Component Llama passe avec ces mêmes limites. Il n’y a pas de branche `if Qwen`.

Le problème est donc architectural/de nommage : le chemin générique dépend encore d’un helper explicitement Qwen, documenté à partir d’un checkpoint Qwen2.5.

**Correctif attendu :**

```text
qwen_component_runtime_limits
→ inference_component_runtime_limits
```

avec une documentation générique. Si des Components ont des besoins différents à terme, ils doivent être exprimés par une politique générique ou le manifeste, pas par une branche famille.

### MAG-02 — 🟡 P2 — Le guard générique ne couvre pas le registre

`tools/check_generic_facade_family_isolation.py` couvre déjà notamment :

```text
production_model_fixture
ProductionLoadedModel
build_first_native_graphs_from_named_component
LoadedInferenceComponent::load
```

mais pas encore :

```text
register_inference_component_artifact
```

C’est pourquoi `qwen_component_runtime_limits()` peut subsister dans le chemin générique sans faire échouer le gate.

**Correctif attendu :** ajouter cette fonction au périmètre du guard et y interdire les helpers famille-spécifiques hors exceptions test/singleton documentées.

## CI

### SHA actuellement consommé par Tachyon

```text
7b4861613c8da5b1ebf8816bd4e3d66065c9e440
```

Quality entièrement verte :

```text
rustfmt                    ✅
clippy                     ✅
cargo-deny                 ✅
Linux / Windows / macOS    ✅
docs                       ✅
component integration      ✅
model-family isolation     ✅
coverage                   ✅
wasm32 component engine    ✅
wasmtime component engine  ✅
e2e conformance            ✅
provider integration       ✅
format integration         ✅
WIT                        ✅
OpenSpec                   ✅
MSRV                       ✅
```

### Magnetar `main`

```text
7e1be5b23e1931579967701ba789f60d585bec09
```

Quality verte, GPU Runner Smoke Test vert.

## Diff avec Tachyon

Les 5 commits d’écart incluent :

```text
fc310bc  Fix #75: declare Artifact bundle format instead of filesystem heuristics (#90)
89cbd2c  Fix coverage ratchet regression from #75 (#91)
72394c1  chore(deps): update install-action digest (#85)
f1d0dae  chore(deps): update nvidia/cuda tag (#84)
7e1be5b  fix(deps): update wasmtime to v49 (#87)
```

Le commit #90 modifie réellement le contrat Artifact et doit être intégré consciemment côté Tachyon.

## Scope correctif

- [ ] Renommer `qwen_component_runtime_limits()` en helper générique.
- [ ] Réécrire sa documentation comme politique Component générique.
- [ ] Vérifier qu’aucune valeur ne dépend sémantiquement de Qwen.
- [ ] Ajouter `register_inference_component_artifact()` au guard générique.
- [ ] Stabiliser le SHA final à consommer par Tachyon.
- [ ] Conserver la Quality verte sur ce SHA.

## Critères de GO

| Critère | État |
|---|:---:|
| Component explicite reçu | ✅ |
| Trust séparé | ✅ |
| Cache-hit safe | ✅ |
| Component autorité du graphe | ✅ |
| Plusieurs Components | ✅ |
| Generic graph API | ✅ |
| `ProductionLoadedModel` dé-Qwenifié | ✅ |
| Vrai Artifact Llama | ✅ |
| Architecture compatibility gate | ✅ |
| Artifact Format explicite | ✅ |
| `wasm32` / docs / coverage / Quality verts | ✅ |
| Registre générique sans helper `qwen_*` | ❌ |
| Guard couvrant le registre générique | ❌ |

## Verdict

🟢 **GO fonctionnel / 🟠 cleanup architectural mineur**

Les anciens blockers architecture/trust/multi-architecture sont fermés. Le seul reliquat concret identifié dans le chemin générique est `register_inference_component_artifact() → qwen_component_runtime_limits()`, à généraliser et à protéger par le guard.
