## 1. Runtime

- [x] 1.1 `ComponentGraphCache::prepare`: the logits row gather on every prefill graph outside tensor parallelism; the production loop asks `is_prefill_bucket` for the padding rule (`magnetar-runtime` lib tests: 790 pass).

## 2. CUDA Provider

- [x] 2.1 `attention_tiled<PART, VPART>`, `attention_tiled_48_32_kernel` (providers/cuda 758b8ba); `attention_with_a_value_width_matches_reference_cpu` adds 192/128 prefill shapes (passes, run 44374637064).
- [x] 2.2 Grouped `moe-experts` prefill (providers/cuda 90d5efd): `moe_group_*`, `moe_q8_gated_grouped_kernel`, `moe_q8_down_grouped_kernel`, `moe_down_combine_kernel`; `grouped_moe_prefill_matches_the_per_pair_kernels`; `moe_prefill_kernel_times` (diagnostic).
- [x] 2.3 Row blocks (providers/cuda c6434f5): `matmul_split_tcp` 2,048 rows of A per launch, `gated_matmul_split_tcp`; `long_prefill_matmuls_run_in_row_blocks`.
- [x] 2.4 `CudaMultiProvider::free_memory` (diagnostic).
- [x] 2.5 The grouped expert Kernels serve two columns per warp (providers/cuda 49a84d2), each lane converting its quants once per step and reading a pair's values once for both columns.

## 3. GPU qualification

- [x] 3.1 The new conformance tests pass on CUDA, and `moe_prefill_kernel_times` gives the grouped prefill's time against one pair at a time. Run 44375603855: the grouped prefill matches the per-pair Kernels bit for bit, 107.56 against 187.09 ms at 1,024 tokens, 429.70 against 750.54 ms at 4,096; DeepSeek-V2-Lite's TTFT 3,800.4 ms (1,024 tokens) and 16,281.5 ms (4,096). Run 44382825307: every conformance test passes, `long_prefill_matmuls_run_in_row_blocks` included. Run 44389385779: two columns per warp, still bit for bit, 76.17 ms at 1,024 tokens and 302.26 ms at 4,096.
- [x] 3.2 DeepSeek-V2-Lite-Chat Q8_0 across both GPUs (balanced) prefills 16,384 tokens with `f32` and `tq4` latent histories; TTFT and TPOT recorded in `cache-latent-attention-history` task 6.7. Run 44382825307: `tq4` does (TTFT 99.5 s with the exact head, 114.7 s without); `f32` runs out of the second GPU's memory at layer 25.
  - Result, 2026-11-25 (run 45576847703, the `latent` suite, providers/cuda 8182423): the `f32` history now prefills 16,384 tokens on the two 12 GB GPUs, the layers balanced (free memory after the load: 3,890 MiB on the first, 2,740 MiB on the second). Over 64 tokens, every request equal to the `f32` one (it is that one):

    | `f32` latent history | TTFT | TPOT |
    | --- | --- | --- |
    | 4,096 tokens | 13,122 ms | 17.03 ms |
    | 8,192 tokens | 31,069 ms | 21.05 ms |
    | 12,288 tokens | 54,133 ms | 24.95 ms |
    | 16,384 tokens | 82,862 ms | 29.56 ms |

    The out-of-memory of run 44382825307 no longer reproduces: later changes to the prefill (the chunked latent passes and what followed) removed it. With `tq4` at 16,384 tokens (runs 44382825307 and 44389385779): TTFT 82,740 ms with the exact head, 90,385 ms without. The task's two histories both prefill 16,384 tokens, so it closes on its own wording; whether `f32` goes further (32,768 tokens) is tried by the `moe` suite's long-context step.
  - Analysis, 2026-11-25 (the owner prefers failing fast; asked whether it costs the nominal cases anything):
    - The failure is already structured, not a crash: the CUDA allocation failure is `KernelOutOfDeviceMemory`, which the generation ends as a memory-limit finish. What is wrong is when: after 74 s, at layer 25.
    - The history's size is known up front (1,024 MB in `f32`, 144 MB in `tq4`, `kv_set_bytes`) and its stores are admitted at the first step. What does not fit is the prefill's working memory on the second GPU (176 MiB free after the failure, 1,806 MiB on the first).
    - A check costs the nominal cases nothing: one free-memory query per Device at a request's first prefill and a plan's peak live bytes computed once and cached with the prepared plan, outside every decode step. Its risk is false refusals, not time.
    - A generic Runtime check cannot decide this case, though. Which Device runs which layer is the multi-GPU Provider's placement, and refusing against the roomiest Device would not have fired here (1,806 MiB free on the first). The check belongs in the multi-GPU Provider, with per-Device free memory and per-segment peak estimates.
  - Decision (owner, 2026-11-25): follow the proposal below. The task closes with the `f32` ceiling's measurement (`latent` suite of the GPU workflow: contexts 4,096, 8,192, 12,288 and 16,384 with an `f32` history); the early refusal is carried by `enforce-local-multigpu-and-expert-placement` (Inherited items).
  - Proposed to close this task as: `tq4` prefills 16,384 tokens (done), `f32`'s ceiling is documented (4,096 tokens measured; 8,192 to measure), and the early refusal is carried by the multi-GPU Provider work.
