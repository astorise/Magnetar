## Why

`support-gptq-quantized-huggingface-ingestion` closed the GPTQ portion of the original Tachyon scope charter's quantization item; AWQ and BitsAndBytes remained. Both are real, distinct, widely-published Hugging Face/Safetensors-shaped quantization conventions (Qwen and Unsloth both publish official AWQ- and BitsAndBytes-quantized `Qwen2.5-0.5B-Instruct` checkpoints), each with its own real, genuinely fragile packing detail -- AWQ's non-sequential GEMM-kernel nibble interleave, BitsAndBytes' double-quantized non-uniform NF4 codebook -- that a documentation-only port risks getting wrong. Closing both now, verified against real checkpoint bytes the same way GPTQ was, completes the charter's quantization item in full.

## What Changes

- `loaders/huggingface` gains `awq.rs`: dequantizes an AWQ (GEMM-kernel) 4-bit-quantized projection's `qweight`/`qzeros`/`scales` triple into a plain `F32` tensor, using the real `AWQ_REVERSE_ORDER` nibble interleave ported directly from AutoAWQ's own `packing_utils.py` and no zero-point offset (unlike GPTQ's `+1`) -- verified against real bytes from `Qwen/Qwen2.5-0.5B-Instruct-AWQ`.
- `loaders/huggingface` gains `bnb.rs`: dequantizes a BitsAndBytes (NF4/FP4, with or without double quantization) 4-bit-quantized projection into a plain `F32` tensor. Uniquely among the three schemes, a BnB-quantized projection keeps the literal raw `.weight` tensor name (disambiguated by sibling `.absmax`/`.quant_map`/`.nested_absmax`/`.nested_quant_map` tensors and a real JSON `quant_state` blob this ingestor reads at extraction time -- the one case needing to read file bytes ahead of the Runtime's own later payload access), and its dequantized result reconstructs to `nn.Linear`'s own `[out_features, in_features]` storage convention (unlike GPTQ/AWQ's already-`[in_features, out_features]`-oriented output), so it is *not* excluded from this crate's existing projection-weight transpose. Verified against real bytes from `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit`.
- `gptq.rs`'s `extract_gptq_projections` now skips (rather than errors on) a `.qweight` tensor with no `.g_idx` sibling, since AWQ shares the identical `.qweight`/`.qzeros`/`.scales` raw naming with no `.g_idx` of its own -- the real disambiguating signal between the two schemes, letting `awq.rs`'s own extractor claim what GPTQ's leaves untouched.
- `weights.rs`'s `WeightsPayloadSource` enum gains `Awq`/`Bnb` variants alongside `Plain`/`Gptq`.
- **BREAKING**: none. A bundle with none of these tensor shapes ingests exactly as before; this is purely additive.

## Capabilities

### New Capabilities
(none -- extends `huggingface-model-ingestion`, added by the GPTQ change)

### Modified Capabilities
- `huggingface-model-ingestion`: gains two new requirements (AWQ dequantization, BitsAndBytes dequantization) alongside the existing GPTQ one.

## Impact

- `loaders/huggingface/src/awq.rs`, `loaders/huggingface/src/bnb.rs`: new modules.
- `loaders/huggingface/src/gptq.rs`: `extract_gptq_projections` skips (not errors on) a `.qweight` tensor with no `.g_idx`.
- `loaders/huggingface/src/weights.rs`: `WeightsPayloadSource` gains `Awq`/`Bnb` variants; extraction call sites for both, in order after GPTQ.
- `loaders/huggingface/fixtures/awq/`, `loaders/huggingface/fixtures/bnb/`: real checkpoint-derived verification fixtures (~290KB and ~294KB).
- `SUBMODULES.md`: `loaders/huggingface` pin and compatibility-matrix row updated.
