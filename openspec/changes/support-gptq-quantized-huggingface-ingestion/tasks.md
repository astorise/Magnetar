## 1. Investigation

- [x] 1.1 Confirmed real network access is available and downloaded the real public `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` checkpoint's `config.json` (`quant_method: gptq, bits: 4, group_size: 128, sym: true, desc_act: false`) and its `model.safetensors` header (via a small HTTP Range request) to learn the real on-disk tensor shapes for `qweight`/`qzeros`/`scales`/`g_idx`.
- [x] 1.2 Downloaded the real `self_attn.v_proj` GPTQ tensor quadruple and the equivalent unquantized weight from `Qwen/Qwen2.5-0.5B-Instruct` (both via targeted Range requests, not a full checkpoint download), and used them to empirically determine the correct zero-point convention (`quant_code - (zero_code + 1)`) -- confirmed the "+1" offset by comparing both formulas' mean absolute error against the real unquantized weight (0.0014 with it, 0.0048 without: a clear, non-overlapping separation).
- [x] 1.3 Confirmed `loaders/huggingface`'s existing `naming::normalize_tensor_name`/`config::parse` already accept a GPTQ-shaped bundle's real HF tensor-naming/config conventions unmodified for everything except the new `.qweight`/`.qzeros`/`.g_idx` suffixes themselves.

## 2. Implementation

- [x] 2.1 `loaders/huggingface/src/gptq.rs`: real 4-bit dequantization math (`dequantize_gptq_projection`), raw-tensor-quadruple extraction and synthesis (`extract_gptq_projections`), and a lazy payload-source wrapper (`GptqDequantizingPayloadSource`).
- [x] 2.2 `loaders/huggingface/src/weight_layout.rs`: `swap_declared_projection_shapes`/`TransposingPayloadSource::new` gain an `already_oriented` exclusion set so a GPTQ-dequantized projection is not transposed a second time.
- [x] 2.3 `loaders/huggingface/src/weights.rs`: `discover_and_parse_weights` wires GPTQ detection in before the existing tensor-name canonicalization loop, returning the new `WeightsPayloadSource` enum and the GPTQ canonical-name exclusion set.
- [x] 2.4 `loaders/huggingface/src/lib.rs`: `ingest()` threads the exclusion set through to both transpose call sites.
- [x] 2.5 `loaders/huggingface/fixtures/gptq/`: checked in the small (~290KB) real checkpoint-derived verification fixtures.

## 3. Tests

- [x] 3.1 `gptq::tests::dequantize_gptq_projection_matches_a_real_public_checkpoint`: dequantizes the real checked-in fixture bytes and compares every one of 896x128 elements against the real unquantized reference, with thresholds tight enough (mean < 0.0025, max < 0.02) to reject the "+1"-offset bug this test caught during development, not just "does not crash".
- [x] 3.2 `gptq::tests::dequantizes_a_hand_built_two_group_projection`/`rejects_a_bit_width_other_than_four`/`extract_gptq_projections_replaces_the_four_siblings_with_one_placeholder`: unit coverage for the packing math and tensor-quadruple extraction in isolation.
- [x] 3.3 `tests::ingests_a_gptq_quantized_bundle_end_to_end`: full `HuggingFaceIngestor::ingest()` pipeline proof, not just the low-level function -- caught a real wiring bug (`payload_source.locations` overwrite dropping the raw GPTQ sibling locations), fixed before this change closed.
- [x] 3.4 `cargo test` (56 tests, all green), `cargo fmt`, `cargo clippy --all-targets -- -D warnings` all clean for `loaders/huggingface`.

## 4. Documentation

- [x] 4.1 `openspec validate support-gptq-quantized-huggingface-ingestion --strict` passes.
- [x] 4.2 `SUBMODULES.md` updated: `loaders/huggingface` pin and compatibility-matrix row.
- [x] 4.3 README updated once archived: the original scope charter's quantization item gains GPTQ.
