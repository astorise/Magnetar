## Why

`wire-generic-inference-component-runtime` landed a generic, digest-keyed Component registry in `magnetar-runtime`, verified correct, but explicitly deferred wiring it into `inference-components::LoadedInferenceComponent` -- the Tachyon-facing facade the integration audit (`audit-magnetar-integration-tachyon.md`) actually flagged. This change closes that gap: `LoadedInferenceComponent::load` now registers the caller-supplied Component under the caller's own `ComponentTrustStore` and threads the resulting digest all the way to generation, making the registered Component the real execution authority instead of the CLI's single hardcoded `"qwen-test"` singleton.

## What Changes

- `ArtifactTrustPolicy` gains a real, separate `ComponentTrustStore` (`trust_component_digest`), independent of its existing `ModelTrustStore` (`trust_digest`) -- closes MAG-03: trusting a Component binary and trusting a Model Artifact are no longer the same decision.
- `LoadedInferenceComponent::load` calls `register_inference_component_artifact` (the caller's own trust, the artifact's own real digest) instead of `register_qwen_component_artifact` (hardcoded digest, single global slot), and passes the returned digest to a new `ProductionQwenLoadedModel::load_with_component` instead of `load` -- closes MAG-02: the registered Component genuinely drives both plan production and dispatch-time graph production for this loaded instance, not the hardcoded singleton.
- `magnetar-runtime`: `ProductionQwenLoadedModel`/`E2eRuntimeModelExecutionEngine` gain an `Option<ComponentDigest>` field (`None` everywhere except this new path, reproducing every pre-existing behavior exactly); `load_with_component` is the new entry point, `load` becomes a one-line wrapper calling it with `None`.
- **BREAKING**: none. `ProductionQwenLoadedModel::load`'s signature and behavior are unchanged; every existing caller and test is unaffected.

## Capabilities

### New Capabilities
(none -- closes gaps in the existing `inference-components`/Component-loading surface)

### Modified Capabilities
- `component`: adds "Component Trust Is Independent Of Model Artifact Trust" -- an embedder handling both a Component and a Model Artifact SHALL evaluate each trust decision independently, never substituting one for the other. `model-component-graph-contract`'s generic registry (added by `wire-generic-inference-component-runtime`) now has a real production consumer -- an embedder-loaded Component is the execution authority for its own generation, not merely registered inertly.

## Impact

- `magnetar-runtime/src/first_native_runtime.rs`: `ProductionQwenLoadedModel::component_digest` field, `load_with_component`, `E2eRuntimeModelExecutionEngine::component_digest` field (threaded through all 7 construction sites, `None` everywhere but the new path), `prepare_generation`/`execute_generation_step` branch on it.
- `inference-components/src/lib.rs`: `ArtifactTrustPolicy::trust_component_digest`/`component_trust_store`, `load`'s registration/loading call sites updated, `register_component_artifact` (the old single-purpose wrapper around `register_qwen_component_artifact`) removed.
- Tests: `production_qwen_loaded_model_load_with_component_matches_the_singleton_path` (`magnetar-runtime`) -- the load-bearing proof: a model loaded via `load_with_component(..., Some(digest))` generates *identical* tokens to the same model loaded via the pre-existing hardcoded-singleton `load`, for the identical real Component bytes. Full regression (1261 tests in `magnetar-runtime`, zero failures), `clippy --all-targets -- -D warnings` and `fmt --check` clean on both crates.
- **Known, honestly-documented gap** (see `design.md`): `inference-components::LoadedInferenceComponent::load`'s own orchestration (HuggingFace ingestion, trust evaluation, fixture construction, the new registration call) has no dedicated `#[test]` in that crate -- a pre-existing gap this change does not introduce (no test exercised `LoadedInferenceComponent::load` at all before this change either), but one this change's real fix would ideally close with a local-bundle-based integration test, deferred as future work.
