## 1. Implementation

- [x] 1.1 Added `loaders/gguf/src/dequantize.rs`: `dequantize_q8_0`, `dequantize_q4_k`, `dequantize_q5_k`, `get_scale_min_k4`, and a ported `f16_to_f32` -- verbatim copies of `magnetar-runtime`'s already-verified implementations (`support-gguf-quantized-tensor-dequantization`), matching this crate's existing "ported, not shared" convention for externalized-module logic.
- [x] 1.2 Rewrote `weights.rs`: removed the blanket quantized-tensor rejection. `TensorLocation` now carries the original file offset/length, the *declared* (post-dequantization) length, and a `TensorSource` (`Plain`/`Q8_0`/`Q4K`/`Q5K`). `GgufPayloadSource::read_payload` reads the original bytes, verifies a declared digest against them (before conversion, matching `weight_layout.rs`'s existing "verify before transform" precedent), and dequantizes before returning for a quantized source.
- [x] 1.3 `discover_and_parse_weights` overrides a quantized tensor's `storage_dtype` (to `F32`), `size_bytes` (to the dequantized byte count), `quantization` (to `None`), and `digest` (to `None`) at discovery time, before naming/transpose runs -- every later step treats it exactly like a real `F32` tensor.
- [x] 1.4 Updated crate-level and `README.md` documentation to describe the new scope (quantized formats supported, GPTQ/AWQ/BitsAndBytes still out of scope).

## 2. Tests

- [x] 2.1 4 new unit tests in `dequantize.rs` (`f16_to_f32` edge cases, `Q8_0`, `Q4_K` both scale-packing branches, `Q5_K` high-bit plane) -- same hand-verified reference values as `support-gguf-quantized-tensor-dequantization`'s own tests.
- [x] 2.2 `weights.rs`'s `dequantizes_a_q8_0_tensor_and_overrides_its_declared_dtype`: proves a discovered quantized tensor's declared dtype/size/quantization are correctly overridden and its payload reads back as real dequantized `F32`.
- [x] 2.3 `lib.rs`'s `dequantizes_then_transposes_a_non_square_quantized_projection`: the scenario this change exists for -- a hand-constructed, non-square, genuinely `Q8_0`-quantized projection weight, ingested through the full `GgufIngestor::ingest` pipeline, transposes to exactly the hand-computed expected values. Passed on the first run.
- [x] 2.4 Downloaded the real public `Qwen2.5-0.5B-Instruct-GGUF` checkpoint's actual `Q8_0` export (sha256-verified against the real downloaded bytes) and added `tests_gguf_real_checkpoint_smoke.rs`'s `#[ignore]`d `real_public_q8_0_gguf_checkpoint_loads_and_generates_on_reference_cpu`.
- [x] 2.5 Ran it against the real downloaded file: every tensor confirmed dequantized to `F32` by ingestion time; generation produced `" 1000000"` for `"The capital of France is"` -- the *exact same output* as the unquantized F16 GGUF export and the Safetensors counterpart already verified in earlier changes, on real quantized production weights. Real, meaningful evidence beyond "does not crash."
- [x] 2.6 Full regression: `loaders/gguf` (34 tests), `integration-tests/production-loading` (19 passed + 6 `#[ignore]`d), `magnetar-runtime` unaffected (not touched by this change). `cargo fmt --check`/`cargo clippy --all-targets -- -D warnings` clean on every touched crate (also caught and fixed pre-existing fmt drift in `tests_gguf_real_checkpoint_smoke.rs` from the prior GGUF-wiring change, unrelated to this fix).

## 3. Documentation

- [x] 3.1 `README.md` (main repo): updated the GGUF/quantization bullets in "currently supports"/"not yet supported" and the Tachyon scope charter reconciliation note.
- [x] 3.2 `openspec validate resolve-gguf-quantized-projection-transpose-sequencing --strict` passes.
