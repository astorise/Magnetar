## 1. Investigation

- [x] 1.1 Downloaded the real public `Qwen/Qwen2.5-0.5B-Instruct-AWQ` checkpoint's `config.json` (`quant_method: awq, version: gemm, bits: 4, group_size: 128, zero_point: true`) and its `model.safetensors` header, learning the real on-disk `qweight`/`qzeros`/`scales` shapes -- no `g_idx`, the real disambiguating signal against GPTQ.
- [x] 1.2 Fetched AutoAWQ's real `awq/utils/packing_utils.py` directly (not recalled from memory) to get the exact `AWQ_REVERSE_ORDER` nibble-interleave convention and confirmed it empirically against real downloaded `self_attn.v_proj` tensor bytes compared to the equivalent unquantized weight from `Qwen/Qwen2.5-0.5B-Instruct`: the real reorder gives mean error 0.00376 (full-tensor, 114688 elements) vs. 0.00899 for the naive sequential order.
- [x] 1.3 Downloaded the real public `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` checkpoint's `config.json` (`quant_method: bitsandbytes, bnb_4bit_quant_type: nf4, bnb_4bit_use_double_quant: true`) and its `model.safetensors` header, discovering the real double-quantized tensor shape (`weight`/`weight.absmax`/`weight.nested_absmax`/`weight.nested_quant_map`/`weight.quant_map`/`weight.quant_state.bitsandbytes__nf4`) and fetched the real `quant_state` JSON blob directly to learn its exact fields (`blocksize`, `shape`, `nested_blocksize`, `nested_offset`).
- [x] 1.4 Confirmed the real nibble order and double-dequantization formula empirically against real downloaded `self_attn.v_proj` bytes: the correct order gives mean error 0.00085 (full-tensor, 114688 elements) vs. 0.01246 for the wrong order.

## 2. Implementation

- [x] 2.1 `loaders/huggingface/src/awq.rs`: real AWQ dequantization (`dequantize_awq_projection`), tensor-triple extraction (`extract_awq_projections`), and a lazy payload-source wrapper (`AwqDequantizingPayloadSource`).
- [x] 2.2 `loaders/huggingface/src/gptq.rs`: `extract_gptq_projections` now skips (`continue`), rather than errors on, a `.qweight` tensor with no `.g_idx` sibling.
- [x] 2.3 `loaders/huggingface/src/bnb.rs`: real BitsAndBytes dequantization (`dequantize_bnb_projection`, single- and double-quantized), extraction that mutates the packed weight tensor's metadata in place (`extract_bnb_projections`), and a lazy payload-source wrapper (`BnbDequantizingPayloadSource`).
- [x] 2.4 `loaders/huggingface/src/weights.rs`: `WeightsPayloadSource` gains `Awq`/`Bnb` variants; `discover_and_parse_weights` runs GPTQ, then AWQ, then BnB extraction, in that order, before the renaming loop.
- [x] 2.5 `loaders/huggingface/fixtures/awq/`, `loaders/huggingface/fixtures/bnb/`: checked in the real checkpoint-derived verification fixtures.

## 3. Tests

- [x] 3.1 `awq::tests::dequantize_awq_projection_matches_a_real_public_checkpoint`/`bnb::tests::dequantize_bnb_projection_matches_a_real_public_checkpoint`: full-tensor comparisons against the real checked-in fixtures, thresholds tight enough to reject the wrong-order bugs found during development.
- [x] 3.2 Unit coverage for both modules' packing math and tensor extraction in isolation (hand-built fixtures).
- [x] 3.3 `tests::ingests_an_awq_quantized_bundle_end_to_end`/`tests::ingests_a_bnb_quantized_bundle_end_to_end`: full `HuggingFaceIngestor::ingest()` pipeline proofs. The BnB one caught a real wiring bug (the packed weight tensor's location ends up under its canonical, not raw, key after renaming -- `extract_bnb_projections`'s own `weight_range.identity` fix), fixed before this change closed.
- [x] 3.4 `gptq::tests::extract_gptq_projections_skips_a_qweight_tensor_with_no_g_idx_sibling`: locks in the new skip-not-error behavior.
- [x] 3.5 `cargo test` (67 tests, all green), `cargo fmt`, `cargo clippy --all-targets -- -D warnings` all clean for `loaders/huggingface`.

## 4. Documentation

- [x] 4.1 `openspec validate support-awq-and-bitsandbytes-quantized-huggingface-ingestion --strict` passes.
- [x] 4.2 `SUBMODULES.md` updated: `loaders/huggingface` pin and compatibility-matrix row.
- [x] 4.3 README updated once archived: the original scope charter's quantization item is fully closed (GPTQ, AWQ, BitsAndBytes all real).
