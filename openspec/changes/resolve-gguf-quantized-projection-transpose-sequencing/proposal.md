## Why

`support-gguf-quantized-tensor-dequantization` made `Q8_0`/`Q4_K`/`Q5_K` dequantization real at the Model Loading layer, but explicitly identified a real blocker preventing `loaders/gguf` from actually using it: the loader's projection-weight transpose operates on raw bytes assuming a flat per-element byte width, which is meaningless for block-quantized data and would silently corrupt it if applied directly. Closing this is what makes a real, genuinely quantized GGUF checkpoint -- the single most common real-world GGUF distribution shape -- actually loadable end to end for the first time.

## What Changes

- `loaders/gguf` gains its own copy of the `Q8_0`/`Q4_K`/`Q5_K` dequantization algorithm (`src/dequantize.rs`, the same real, upstream-verified formulas `support-gguf-quantized-tensor-dequantization` already ported into `magnetar-runtime`), applied at payload-read time *before* the existing projection-weight transpose runs.
- A quantized tensor's declared `storage_dtype`/`size_bytes`/`quantization` are overridden to reflect the dequantized `F32` reality once discovered, so every later step in the loader's pipeline (transpose, tied-`lm_head` derivation, manifest construction) treats it exactly like a real `F32` tensor -- no special-casing needed downstream, and `magnetar-runtime`'s own generic dequantization is never redundantly reached for a GGUF-sourced tensor.
- The blanket rejection of quantized tensors is removed.
- **BREAKING**: none. A GGUF file with only unquantized tensors behaves identically to before.

## Capabilities

### New Capabilities
(none)

### Modified Capabilities
- `gguf-model-ingestion`: removes the "GGUF Ingestion Rejects Quantized Tensors Structurally" requirement, superseded by a new "GGUF Ingestion Dequantizes Supported Block Formats" requirement covering the three block formats `formats/gguf` already recognizes; GPTQ/AWQ/BitsAndBytes (unrelated to GGUF's block format) remain out of scope and continue to fail at the parser layer (`formats/gguf` never recognizes their `ggml_type` values in the first place).

## Impact

- `loaders/gguf`: new `src/dequantize.rs` (ported from `magnetar-runtime`, including its own `f16_to_f32` and full test coverage); `src/weights.rs` rewritten to dequantize-then-serve for a quantized tensor instead of rejecting it.
- Tests: 4 new unit tests in `dequantize.rs` (mirroring `support-gguf-quantized-tensor-dequantization`'s own hand-verified reference values), 1 new `weights.rs` test proving a quantized tensor's declared dtype is overridden correctly, 1 new `lib.rs` end-to-end test proving a hand-constructed, non-square, genuinely quantized projection weight dequantizes *and* transposes to the exact expected values (the scenario this whole change exists for), 1 new real-checkpoint test against the actual public `Qwen2.5-0.5B-Instruct-GGUF` `Q8_0` export.
- **Verified against a real, genuinely quantized public checkpoint**: the real `Q8_0` export of `Qwen2.5-0.5B-Instruct-GGUF` now loads and generates end to end, producing the exact same output as its unquantized F16 GGUF export and its Safetensors counterpart for the same prompt -- strong, real-world evidence (not merely "does not crash") that the dequantize-then-transpose sequencing is correct.
