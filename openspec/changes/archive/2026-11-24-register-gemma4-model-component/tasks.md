## 1. Artifact

- [x] 1.1 Pin `components/gemma` at `9186a0c`; describe it in `SUBMODULES.md`.
- [x] 1.2 Build it with the pinned toolchain (`wasm32-unknown-unknown`, release, `wasm-tools component new`); two builds give the same digest.
- [x] 1.3 Check in `gemma-real.component.wasm` and its manifest: WIT `1.6.0` imports, families `gemma4`, `gemma4_text`, `gemma4_unified`, formats `huggingface` and `gguf`.

## 2. Tests

- [x] 2.1 Engine tests in `magnetar-engine` (from the Component's `docs/engine-smoke-test/`), with pinned node counts:
  - dense, MoE and per-layer-input layouts;
  - decode and segments, including the refused shared-KV split;
  - the 31B, 26B-A4B and E2B shapes.
- [x] 2.2 Logit comparison with `transformers` in `integration-tests/production-loading` (from `tools/qualification/`): two Hugging Face bundles and four quantized GGUF files with their references; `gemma-real` among the real Components.
- [x] 2.3 A negative control: the quantized file against the unquantized reference fails.

## 3. Validation

- [x] 3.1 `cargo fmt`, `clippy -D warnings` and the tests of `magnetar-engine` and of the production-loading crate.
