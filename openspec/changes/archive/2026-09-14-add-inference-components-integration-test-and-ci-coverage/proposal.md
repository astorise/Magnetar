## Why

The Tachyon integration audit (`docs/audits/audit-magnetar-integration-tachyon-2026-09-13.md`) reviewed `inference-components::LoadedInferenceComponent` -- the crate's own Tachyon-facing boundary -- and its closure notes left one gap on record: no integration test drives `LoadedInferenceComponent::load` itself end to end (only the equivalent `magnetar-runtime` path underneath it was tested). Building that test surfaced a second, more serious gap while wiring it up: `inference-components` is its own `[workspace]` (like `magnetar-cli`, moved out of the repository root because it depends on `loaders/gguf`/`loaders/huggingface`/`providers/cpu` submodules), but unlike `magnetar-cli` it was never added to the one CI job (`submodule-integration`) that checks those submodules out and builds/tests/lints crates depending on them. The crate the audit reviewed most closely has been built and tested by nobody, in CI, at any point.

## What Changes

- `inference-components/src/lib.rs` gains a real end-to-end integration test module (`load_end_to_end_tests`): a completely-specified, real Hugging Face-shaped bundle (every tensor `magnetar_runtime::qwen_expected_tensor_shape` requires, at the correct pre-transpose shapes) is written to a temp directory and loaded through `LoadedInferenceComponent::load` with the real, checked-in Qwen Component fixture, then driven through a real generation via `invoke_payload` -- the same call shape a real embedder uses. A second test proves Model Artifact trust rejection is still enforced from this crate's own public entry point, not only at the `magnetar-runtime` layer underneath it.
- `inference-components/Cargo.toml` gains a `[dev-dependencies]` `tempfile = "3"` (matching `loaders/huggingface`'s and `loaders/gguf`'s own dev-dependency).
- `.github/workflows/quality.yml`'s `submodule-integration` job gains `inference-components/Cargo.toml` in its test loop, plus dedicated format-check and clippy steps mirroring `magnetar-cli`'s own treatment -- closing the CI-coverage gap found above.
- **BREAKING**: none. Test-only additions and CI-only workflow changes; no production code path changes.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `quality`: "Repository Continuous Integration" gains a scenario stating that a first-party crate moved out of the root workspace because it depends on externalized submodules (the `magnetar-cli` precedent) SHALL still be covered by CI, not silently left untested -- closing the gap found here for `inference-components`.

## Impact

- `inference-components/src/lib.rs`: new `#[cfg(test)] mod load_end_to_end_tests`, no production code changes.
- `inference-components/Cargo.toml`, `inference-components/Cargo.lock`: new dev-dependency.
- `.github/workflows/quality.yml`: `submodule-integration` job now builds, tests, format-checks, and lints `inference-components`.
- README/`docs/audits/audit-magnetar-integration-tachyon-2026-09-13.md`: the closure note's remaining "no dedicated integration test" gap is closed; the audit's own "Suivi de clôture" section records the CI-coverage finding.
