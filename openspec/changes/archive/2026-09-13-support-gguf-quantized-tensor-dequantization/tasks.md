## 1. Research

- [x] 1.1 Verified `block_q8_0`/`block_q4_K`/`block_q5_K`'s exact byte layouts and dequantization formulas directly against real, downloaded `ggml-org/llama.cpp` source (`ggml-common.h`, `ggml-quants.c`), cross-checking `get_scale_min_k4`, `dequantize_row_q4_K`, and `dequantize_row_q5_K` byte-for-byte -- not recalled from general knowledge.

## 2. Implementation

- [x] 2.1 Added `dequantize_q8_0`, `dequantize_q4_k`, `dequantize_q5_k`, and shared `get_scale_min_k4` to `magnetar-runtime/src/model_loading.rs`.
- [x] 2.2 Generalized `bytes_per_element(tensor) -> u64` to `expected_storage_bytes(element_count, tensor) -> u64`, block-aware (validates element count divides evenly by the format's block size before computing total bytes).
- [x] 2.3 Wired `ModelDType::Q8`/`Q4K`/`Q5K` into `tensor_from_raw_bytes`'s dtype-decode match, alongside the existing `F32`/`F16`/`BF16` arms.
- [x] 2.4 Extended `WeightMaterializationTransaction::stage_weight`'s declared-dtype whitelist (`first_native_runtime.rs`) to accept `Q8`/`Q4K`/`Q5K`, matching `F16`/`BF16`'s existing acceptance of already-converted `F32` content; updated the now-stale comment above it.
- [x] 2.5 Updated `host_tensors_from_artifact_bytes`'s doc comment to describe the new supported dtypes and digest-check scope.

## 3. Tests

- [x] 3.1 `host_tensors_from_artifact_bytes_dequantizes_q8_0`: a 34-byte block with scale 2.0 and a -16..15 ramp, verified against the exact `value[i] = d * qs[i]` formula.
- [x] 3.2 `host_tensors_from_artifact_bytes_dequantizes_q4_k`: a hand-constructed 144-byte block exercising both `get_scale_min_k4` branches (sub-blocks 0/1 direct, sub-blocks 4/5 split-byte) and the low/high-nibble-across-two-sub-blocks-per-64-chunk interleaving. **A real hand-calculation error was caught here**: an earlier version of this test's own expected value for sub-block 1 omitted the quantized nibble value as a multiplicative factor (computed `d_sb - m_sb` instead of `d_sb * nibble - m_sb`); the test failing against the (already-correct) implementation caught it, not the other way around -- fixed by correcting the test, not the implementation.
- [x] 3.3 `host_tensors_from_artifact_bytes_dequantizes_q5_k`: isolates the `qh` high-bit-plane reconstruction specifically (all other sub-blocks zeroed so only the one exercised bit position needs hand-verification).
- [x] 3.4 `materialize_model_instance_weights_accepts_dequantized_content_for_a_quantized_declared_dtype`: end-to-end acceptance across all three dtypes at the `materialize_model_instance_weights` level (not just the lower-level `tensor_from_raw_bytes`), proving `stage_weight`'s whitelist fix took effect.
- [x] 3.5 Updated two pre-existing tests (`host_tensors_from_artifact_bytes_rejects_unsupported_dtype`, `materialize_model_instance_weights_rejects_quantized_declared_dtype`) that asserted `ModelDType::Q8` was rejected -- no longer true -- to use `ModelDType::I8` instead, preserving their original "a genuinely unsupported dtype is still rejected" intent.
- [x] 3.6 Full regression: `cargo test -p magnetar-runtime --all-features` (1254 + 173 passed), `cargo fmt --check`/`cargo clippy --all-targets --all-features -- -D warnings` clean.

## 4. Scope decision (documented, not implemented)

- [x] 4.1 Investigated wiring quantized tensor support into `loaders/gguf` and found a real architectural blocker: its projection-weight transpose operates on raw bytes assuming a flat per-element width, which is incompatible with block-quantized data and would silently corrupt it. Documented in design.md as a real, identified, deliberately-deferred follow-up rather than working around it or silently shipping something wrong.
