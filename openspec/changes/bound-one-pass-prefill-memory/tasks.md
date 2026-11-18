## 1. Runtime

- [x] 1.1 `ComponentGraphCache::prepare`: the logits row gather on every prefill graph outside tensor parallelism; the production loop asks `is_prefill_bucket` for the padding rule (`magnetar-runtime` lib tests: 790 pass).

## 2. CUDA Provider

- [x] 2.1 `attention_tiled<PART, VPART>`, `attention_tiled_48_32_kernel` (providers/cuda 758b8ba); `attention_with_a_value_width_matches_reference_cpu` adds 192/128 prefill shapes (passes, run 44374637064).
- [x] 2.2 Grouped `moe-experts` prefill (providers/cuda 90d5efd): `moe_group_*`, `moe_q8_gated_grouped_kernel`, `moe_q8_down_grouped_kernel`, `moe_down_combine_kernel`; `grouped_moe_prefill_matches_the_per_pair_kernels`; `moe_prefill_kernel_times` (diagnostic).
- [x] 2.3 Row blocks (providers/cuda c6434f5): `matmul_split_tcp` 2,048 rows of A per launch, `gated_matmul_split_tcp`; `long_prefill_matmuls_run_in_row_blocks`.
- [x] 2.4 `CudaMultiProvider::free_memory` (diagnostic).

## 3. GPU qualification

- [ ] 3.1 The new conformance tests pass on CUDA, and `moe_prefill_kernel_times` gives the grouped prefill's time against one pair at a time.
- [ ] 3.2 DeepSeek-V2-Lite-Chat Q8_0 across both GPUs (balanced) prefills 16,384 tokens with `f32` and `tq4` latent histories; TTFT and TPOT recorded in `cache-latent-attention-history` task 6.7.
