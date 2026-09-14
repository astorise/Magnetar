## 1. Investigation

- [x] 1.1 Read `LoadedInferenceComponent::load` end to end to identify every real artifact an integration test needs: a real Component (reused from the checked-in `qwen-real.component.wasm` fixture), a real bundle on disk (config.json/tokenizer.json/model.safetensors), and the exact tensor names/shapes the real Qwen Component's graph resolves (`magnetar_runtime::qwen_expected_tensor_shape`).
- [x] 1.2 Confirmed `inference-components` is its own `[workspace]`, structurally identical to `magnetar-cli`, and -- unlike `magnetar-cli` -- absent from `submodule-integration`, the one CI job that checks out and builds/tests/lints crates depending on `loaders/gguf`/`loaders/huggingface`/`providers/cpu`. Confirmed via `grep -rln "inference-components" .github/` returning nothing before this change.

## 2. Implementation

- [x] 2.1 `inference-components/src/lib.rs`: added `load_end_to_end_tests` with a hand-built, completely-specified Hugging Face-shaped bundle and two tests -- a full `load` -> `invoke_payload` generation, and a Model Artifact trust rejection from this crate's own public entry point.
- [x] 2.2 `inference-components/Cargo.toml`: added `[dev-dependencies] tempfile = "3"`.
- [x] 2.3 `.github/workflows/quality.yml`: added `inference-components/Cargo.toml` to `submodule-integration`'s test loop, plus dedicated format-check and clippy steps mirroring `magnetar-cli`'s own.

## 3. Tests

- [x] 3.1 `cargo test --manifest-path inference-components/Cargo.toml` run 3 consecutive times locally, all green (5 tests: 3 pre-existing, 2 new).
- [x] 3.2 `cargo fmt --manifest-path inference-components/Cargo.toml --all -- --check` and `cargo clippy --locked --manifest-path inference-components/Cargo.toml --all-targets -- -D warnings` both clean, run exactly as the new CI steps will run them.
- [x] 3.3 `cargo test --locked --manifest-path inference-components/Cargo.toml` (the exact command the CI loop runs) green.

## 4. Documentation

- [x] 4.1 `openspec validate add-inference-components-integration-test-and-ci-coverage --strict` passes.
- [x] 4.2 README/`docs/audits/audit-magnetar-integration-tachyon-2026-09-13.md` updated once archived: the remaining integration-test gap is closed, and the CI-coverage finding is recorded.
