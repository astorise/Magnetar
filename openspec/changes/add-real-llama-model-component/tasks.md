## 1. Investigation

- [x] 1.1 Confirmed `components/llama` was an empty `cargo new --lib` scaffold, pinned as a submodule but never implemented, "not yet scheduled against any OpenSpec change" per its own README.
- [x] 1.2 Read `components/qwen/src/lib.rs` and its WIT contract (`magnetar:model-component-graph@1.2.0`) in full: confirmed the graph-building logic is already architecture-generic (every dimension and the QKV-bias/tied-embeddings conditionals come from `model-config`, none compiled in), and that Llama and Qwen2 are real instances of the same generic pre-norm/RoPE/GQA/SwiGLU decoder block.
- [x] 1.3 Confirmed `loaders/huggingface`'s `normalize_tensor_name`/`config.rs` already accept a Llama-shaped bundle unmodified (identical real Hugging Face tensor-naming convention; bias tensors simply absent from a real Llama checkpoint) -- no loader change needed for this change's scope.

## 2. Implementation

- [x] 2.1 `components/llama`: implemented the real Component (WIT contract copied verbatim from `components/qwen`, graph-building logic structurally identical, documented as a deliberate consequence of real architectural fact). Built via `cargo +1.98.1-x86_64-pc-windows-msvc build --target wasm32-unknown-unknown --release`, then `wasm-tools component new`. Validated (`wasm-tools validate` -> VALID) and confirmed its WIT world is structurally identical to the real Qwen Component's (`wasm-tools component wit` on both, byte-identical output). Committed and pushed to the submodule repository (commit `70692e2`).
- [x] 2.2 Pinned `components/llama` at `70692e2` in this repository; updated `SUBMODULES.md`'s module table row and added a compatibility-matrix row.
- [x] 2.3 `magnetar-runtime/fixtures/components/llama-real.component.wasm` (+ manifest, mirroring `qwen-real.component.wasm`'s own format) added as a checked-in test fixture.
- [x] 2.4 `magnetar-runtime/src/first_native_runtime.rs`: added `LLAMA_REAL_COMPONENT_BYTES`/`LLAMA_REAL_COMPONENT_MANIFEST_BYTES` `#[cfg(test)]` constants, mirroring the existing `QWEN_REAL_COMPONENT_BYTES` pattern.

## 3. Tests

- [x] 3.1 Added `build_first_native_graphs_from_named_component_serves_a_real_second_architecture_family`: registers the real Llama Component under its own real digest, proves it produces byte-identical graphs to the Qwen singleton for the same (bias-free) config, then proves the same Llama binary produces a genuinely different graph for a bias-bearing config.
- [x] 3.2 Hit and fixed a real gap while writing this test: `qwen_operator_kind_code` (test-only) had no entry for the `"add"` bias operator -- `GraphValidationFailed { reason: "... operator 'add' with no known kind code" }` on first run. Fixed by adding `"add" => Some(9)`, purely additive.
- [x] 3.3 `cargo test -p magnetar-runtime --features wasmtime-component-engine --lib` run 3 consecutive times, all green (1262 tests).
- [x] 3.4 `cargo fmt`, `cargo clippy -p magnetar-runtime --features wasmtime-component-engine --tests -- -D warnings`, `cargo check -p magnetar-runtime --target wasm32-unknown-unknown --all-features` all clean.

## 4. Documentation

- [x] 4.1 `openspec validate add-real-llama-model-component --strict` passes.
- [x] 4.2 `components/llama/README.md` updated from "Empty template" to a real-status description.
- [x] 4.3 README/`SUBMODULES.md` updated once archived.
