## Why

The original Tachyon scope charter names GPTQ/AWQ/BitsAndBytes as future quantization work, distinct from GGUF's own quantization (already real via `loaders/gguf`). GPTQ is the most standardized and widely-published of the three real-world formats a Hugging Face-style bundle actually ships (a real, quantized 4-bit `Qwen2.5-0.5B-Instruct-GPTQ-Int4` checkpoint is publicly available from the same `Qwen` publisher this repository already treats as its real-checkpoint reference). Closing it first, verified against real checkpoint bytes rather than documentation alone, gives `loaders/huggingface` its first real quantization support and establishes the same "dequantize before transpose" pattern `loaders/gguf` already proved out for GGUF's own block formats.

## What Changes

- `loaders/huggingface` gains a new `gptq.rs` module: dequantizes a GPTQ-quantized projection's four raw tensors (`qweight`/`qzeros`/`scales`/`g_idx`) into a plain `F32` tensor at ingestion time, before this crate's existing projection-weight transpose runs -- mirroring `loaders/gguf`'s own dequantize-before-transpose precedent for its block formats.
- `weight_layout.rs`'s transpose (`swap_declared_projection_shapes`/`TransposingPayloadSource`) gains an exclusion set: a GPTQ-dequantized projection is already `[in_features, out_features]` by construction (GPTQ's own `qweight` packs the input dimension first), so it must not be transposed a second time.
- The packing convention -- including GPTQ's well-known "+1" zero-point offset -- is verified against real bytes downloaded via targeted HTTP Range requests from the public `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` checkpoint, compared element-by-element against the same real weight's own unquantized value from `Qwen/Qwen2.5-0.5B-Instruct`. Small fixture files (~290KB total) are checked in so this verification runs offline in CI.
- Scoped to 4-bit GPTQ only (the overwhelmingly common real-world case); 2/3/8-bit variants are explicitly rejected with a structured error rather than silently mis-decoded, matching `loaders/gguf`'s own "support one well-verified format first" precedent.
- **BREAKING**: none. A bundle with no `.qweight` tensors ingests exactly as before; this is purely additive.

## Capabilities

### New Capabilities
- `huggingface-model-ingestion`: this crate's first dedicated capability spec, scoped narrowly to what is verified today (GPTQ dequantization), not a full retroactive spec for the whole crate.

## Impact

- `loaders/huggingface/src/gptq.rs`: new module (dequantization math + tensor-quadruple extraction + a lazy payload-source wrapper).
- `loaders/huggingface/src/weight_layout.rs`: `swap_declared_projection_shapes`/`TransposingPayloadSource::new` gain an `already_oriented: &BTreeSet<String>` parameter.
- `loaders/huggingface/src/weights.rs`: `discover_and_parse_weights` wires GPTQ detection in before tensor-name canonicalization; fixes a real bug found by this change's own end-to-end test (`payload_source.locations` was being overwritten instead of extended, silently dropping the raw GPTQ sibling tensors' locations).
- `loaders/huggingface/fixtures/gptq/`: real checkpoint-derived verification fixtures (~290KB).
- `SUBMODULES.md`: `loaders/huggingface` pin and compatibility-matrix row updated.
