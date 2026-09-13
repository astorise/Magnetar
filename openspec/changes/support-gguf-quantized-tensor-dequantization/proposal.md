## Why

Quantization is the next item on the Tachyon scope charter after GGUF wiring closed, and it is the most directly connected: most real-world GGUF checkpoints people actually download are quantized (Q4_K_M, Q8_0, ...), and `formats/gguf` already recognizes these tensors' quantization metadata -- Model Loading simply had nowhere to route them, rejecting every quantized tensor structurally. Closing the Model Loading side of this (dequantization-at-load into F32, exactly mirroring how F16/BF16 storage already works) is real, valuable, and low-risk on its own, independent of whether any specific loader crate is ready to route quantized tensors through it yet.

## What Changes

- `magnetar-runtime`'s Model Loading gains real dequantization for GGUF's three K-quant/block formats -- `Q8_0`, `Q4_K`, `Q5_K` (`ModelDType::Q8`/`Q4K`/`Q5K`) -- converted to `F32` at weight materialization time, in both the whole-buffer (`host_tensors_from_artifact_bytes`) and streaming (`host_tensor_from_payload_source`) paths, exactly like `F16`/`BF16` conversion already works: storage dtype and compute dtype remain distinct, conversion is explicit, and every kernel downstream only ever sees `F32` content.
- The dequantization algorithms are ported bit-for-bit from `ggml-org/llama.cpp`'s real `ggml-quants.c`, verified directly against the current upstream source (not recalled from memory), including the historically error-prone 6-bit sub-block scale/min packing shared by `Q4_K`/`Q5_K`.
- `WeightMaterializationTransaction::stage_weight`'s declared-dtype whitelist is extended to accept `Q8`/`Q4K`/`Q5K` as valid declared storage dtypes backed by already-dequantized `F32` content, matching `F16`/`BF16`'s existing acceptance.
- **Explicitly not included, and clearly flagged as a real, identified blocker rather than silently deferred**: wiring this into `loaders/gguf` so a real quantized GGUF file can actually be ingested end to end. `loaders/gguf`'s existing projection-weight transpose (`weight_layout.rs`) operates on raw bytes assuming a flat per-element byte width; a block-quantized tensor's bytes have no such flat width, so naively removing `loaders/gguf`'s current quantization rejection would silently corrupt any quantized projection weight's block structure during transpose. This needs the transpose to happen *after* dequantization (inside the loader, which would need its own copy of this dequantization logic to sequence correctly), not before -- a real, separate, well-scoped follow-up, not something to ship half-working.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `model-loading`: extends the existing "Quantization Handling" requirement with real dequantize-at-load behavior for supported GGUF block formats, alongside its existing "reject unsupported quantization" scenario (unchanged). `production-model-ingestion` is unaffected -- no loader wiring in this change.

## Impact

- `magnetar-runtime/src/model_loading.rs`: new dequantization functions (`dequantize_q8_0`, `dequantize_q4_k`, `dequantize_q5_k`, shared `get_scale_min_k4`), `bytes_per_element` generalized to `expected_storage_bytes` (block-aware sizing), updated doc comments.
- `magnetar-runtime/src/first_native_runtime.rs`: `stage_weight`'s declared-dtype whitelist extended.
- Tests: 4 new unit tests with hand-computed, independently-verified reference values (`Q8_0`, `Q4_K` exercising both scale/min packing branches, `Q5_K` isolating the high-bit-plane reconstruction, plus an end-to-end `materialize_model_instance_weights` acceptance test across all three dtypes); 2 pre-existing tests updated (they asserted `Q8` was rejected, which is no longer true -- moved to a genuinely still-unsupported dtype to preserve their original intent).
- No breaking changes: every existing `F32`/`F16`/`BF16` code path and test is unaffected.
