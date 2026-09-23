# Audit complet Magnetar — 22 septembre 2026

**Repository :** `astorise/Magnetar`  
**SHA consommé par Tachyon :** `7b4861613c8da5b1ebf8816bd4e3d66065c9e440`  
**Magnetar `main` observé :** `89cbd2c9b29d73e449847893346cf0e6a4277f1a`  
**Verdict sur le SHA intégré :** 🟢 **GO architecture / qualité**  
**Sujet restant :** 🟠 convergence vers `main`

## 1. Invariant

Magnetar reçoit un Component WASM explicite et doit pouvoir gérer :

- Model Artifact ;
- tokenizer ;
- ModelInstance ;
- Providers / Devices / Kernels ;
- prefill/decode/KV ;
- quantization et mécanismes d’exécution ;
- le Component comme autorité réelle de graphe/inférence.

La façade utilisée par Tachyon doit rester indépendante d’une famille de modèle particulière.

## 2. Résultat

Les anciens blocages architecturaux identifiés lors du précédent audit sont fermés sur le SHA actuellement intégré par Tachyon :

```text
7b4861613c8da5b1ebf8816bd4e3d66065c9e440
```

Aucun nouveau P0/P1 architectural n’a été identifié sur ce SHA.

La Quality CI de ce SHA est entièrement verte.

## 3. Anciens findings fermés

| Ancien finding | État |
|---|:---:|
| façade `ProductionLoadedModel` Qwen-shaped | ✅ |
| `QwenConfig` dans le state machine générique | ✅ |
| `production_qwen_component_identity()` générique | ✅ |
| `qwen_validate_model_artifact()` dans la façade | ✅ |
| second Component avec Artifact Qwen réutilisé | ✅ |
| build `wasm32` cassé | ✅ |
| rustdoc rouge | ✅ |
| coverage ratchet rouge | ✅ |
| Component cache trust bypass | ✅ |
| trust Component / Artifact confondu | ✅ |
| Component WASM non autoritaire | ✅ |
| graph API Qwen-spécifique | ✅ |

## 4. Façade générique dé-Qwenifiée

`ProductionLoadedModel` existe toujours, mais les dépendances bloquantes ont disparu du chemin générique.

Ne sont plus présents dans ce chemin :

```text
QwenConfig
production_qwen_component_identity
qwen_component_descriptor
qwen_validate_model_artifact
load_production_qwen_instance_for_provider
QwenLayerKvMap
```

`production_model_fixture()` part maintenant de :

```rust
manifest.architecture_config
```

et de la vraie famille :

```rust
manifest.architecture.family
```

L’identité générique est :

```text
production-model
```

et non plus `production-qwen`.

✅ Ancien P0 fermé.

## 5. Compatibilité Component ↔ architecture

`ProductionLoadedModel::load_with_component(...)` compare désormais :

```rust
registered.supported_architecture_families
```

avec :

```rust
fixture.manifest.architecture.family
```

Un Component Llama ne peut donc plus être associé silencieusement à un Artifact Qwen incompatible.

✅ Correct.

## 6. Vrai second modèle end-to-end

Le test :

```rust
loaded_inference_component_load_runs_a_real_second_architecture_end_to_end()
```

utilise désormais :

```rust
write_tiny_llama_bundle(...)
```

Le fixture Llama possède sa propre configuration, des dimensions distinctes, un vrai schéma GQA différent, deux decoder layers, son propre jeu de tensors, son propre tokenizer, son propre Component WASM et son propre digest.

La preuve est donc maintenant réellement :

```text
Qwen Component + Qwen Artifact
          │
          ├── même façade Magnetar
          │
Llama Component + Llama Artifact
```

✅ Ancien MAG-02 fermé.

## 7. APIs d’invocation opaques

La PR #89 ajoute :

```rust
invoke_payload_opaque(...)
invoke_payload_streaming_opaque(...)
```

Tachyon peut ainsi consommer bytes + tags clé/valeur sans dépendre de :

```text
InferenceComponentOutput
GenerationStreamEvent
GenerationUsage
```

✅ Correct.

## 8. Trust

Sur cache hit, le trust est toujours réévalué :

```rust
let trust_decision =
    trust.evaluate(&manifest, &digest);

if trust_decision.status != ComponentTrustStatus::Trusted {
    return Err(...);
}
```

Le cache réutilise la compilation, jamais l’autorisation d’un autre caller.

Le trust Component et le trust Model Artifact restent séparés.

✅ Correct.

## 9. Component comme autorité réelle

Le Component enregistré est réellement utilisé par :

```rust
build_first_native_graphs_from_named_component(...)
```

avec :

```rust
ModelArchitectureConfig
```

Le `component_digest` est propagé de manière cohérente à la préparation et au dispatch.

L’ancien problème « Component WASM enregistré mais génération native Qwen exécutée à la place » n’est pas revenu.

✅ Correct.

## 10. Quality CI du SHA intégré

SHA :

```text
7b4861613c8da5b1ebf8816bd4e3d66065c9e440
```

Run :

```text
https://github.com/astorise/Magnetar/actions/runs/35724075515
```

Résultats observés :

| Job | État |
|---|:---:|
| rustfmt | ✅ |
| clippy | ✅ |
| cargo-deny | ✅ |
| check Linux | ✅ |
| test Linux | ✅ |
| check macOS | ✅ |
| test macOS | ✅ |
| check Windows | ✅ |
| test Windows | ✅ |
| docs | ✅ |
| coverage | ✅ |
| component integration | ✅ |
| e2e conformance | ✅ |
| provider integration | ✅ |
| WIT | ✅ |
| model-family isolation | ✅ |
| Wasmtime Component Engine | ✅ |
| wasm32 Component Engine | ✅ |
| MSRV | ✅ |
| format integration | ✅ |

Les anciens failures coverage/docs/wasm32 sont fermés.

## 11. MAG-INT-01 — 🟠 P1 — Écart volontaire avec `main`

Tachyon consomme :

```text
7b4861613c8da5b1ebf8816bd4e3d66065c9e440
```

alors que Magnetar `main` est :

```text
89cbd2c9b29d73e449847893346cf0e6a4277f1a
```

Écart : **2 commits**.

Ils correspondent principalement à :

- PR #90 : déclaration explicite du format d’Artifact ;
- PR #91 : correction du coverage ratchet lié à cette évolution.

La PR #90 introduit notamment :

```text
magnetar-artifact-format.yaml
```

qui nécessite une adaptation des fixtures Tachyon.

### Classification

🟠 P1 de convergence, **pas un défaut architectural du SHA intégré**.

Le couple actuel :

```text
Tachyon 6891f9b...
Magnetar 7b486161...
```

reste cohérent et qualifié.

## 12. Scope demandé à l’équipe Magnetar

Aucun correctif P0/P1 architectural supplémentaire n’est demandé sur le SHA actuellement consommé par Tachyon.

Travail de convergence recommandé :

- [ ] documenter clairement `magnetar-artifact-format.yaml`;
- [ ] fournir un exemple minimal pour les intégrateurs ;
- [ ] accompagner l’adaptation des fixtures Tachyon ;
- [ ] confirmer la compatibilité de l’API opaque après #90/#91 ;
- [ ] permettre le repin Tachyon sur un SHA récent ;
- [ ] relancer la Quality CI sur le SHA finalement intégré.

## 13. Critères de GO

| Critère | État |
|---|:---:|
| Component explicite reçu | ✅ |
| Component trust séparé | ✅ |
| Model Artifact trust séparé | ✅ |
| cache hit réévalue le trust | ✅ |
| Component réel autorité de graphe | ✅ |
| plusieurs Components coexistants | ✅ |
| graph API en `ModelArchitectureConfig` | ✅ |
| façade générique sans `QwenConfig` | ✅ |
| vrai Qwen + vrai Llama E2E | ✅ |
| second Artifact réellement distinct | ✅ |
| `wasm32` vert | ✅ |
| docs verts | ✅ |
| coverage vert | ✅ |
| model-family isolation vert | ✅ |
| Quality complète du SHA intégré | ✅ |
| alignement avec le HEAD actuel | ⚠️ |

## 14. Verdict

### 🟢 GO Magnetar sur le SHA intégré

Sur :

```text
7b4861613c8da5b1ebf8816bd4e3d66065c9e440
```

les anciens blocages sont fermés et la Quality CI est entièrement verte.

Le seul sujet restant est la convergence vers les deux commits plus récents de `main`, notamment le nouveau contrat explicite de format d’Artifact.

Le NO-GO global de l’intégration vient actuellement du côté Tachyon, qui interprète encore certaines metadata d’usage au lieu de les transporter intégralement de façon opaque.
