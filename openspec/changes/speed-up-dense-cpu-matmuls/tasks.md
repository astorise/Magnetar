## 1. Optimized CPU Provider

- [x] 1.1 Compact bfloat16 storage through `write_weight_tensor`; forgotten on replacement; `materialize` and `shape`.
- [x] 1.2 `native::dense`: output-major and input-major kernels, f32 and bfloat16, AVX2/FMA with a scalar fallback, across the pool.
- [x] 1.3 The embedding lookup on a compact table.
- [x] 1.4 Tests:
  - `dense_matmuls_match_the_reference_cpu`;
  - `exact_bfloat16_weights_are_stored_compact`;
  - `matmul_kernels_and_lookups_read_compact_weights`;
  - `dense_matmul_bandwidth` (diagnostic).

## 2. Benchmark

- [x] 2.1 `cpu-native` in the qualification campaign; the benchmark's CPU cell selects it.
- [x] 2.2 Benchmark CPU on Qwen2.5-0.5B, all engines in one job (run 44517360795, AMD EPYC 9V74, 4 vCPUs, providers/cpu 9cbb5a5):

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar `cpu-native` (bfloat16 checkpoint) | 752 ms | 95.8 ms | 43.6 ms |
  | llama-server (FP16 GGUF) | 537 ms | 47.8 ms | 32.3 ms |
  | Ollama (FP16 GGUF, 4 prompt tokens cached) | 446 ms | 27.9 ms | 26.0 ms |

  Before this change the same cell ran 7.49 s per request on `cpu-native` (run 44511061319, 419 ms per token), and 19.9 s on the Reference CPU Provider (run 44485741553). A local profile (4 threads) puts about 12 ms of each 58 ms decode step in the Runtime's per-node dispatch, which the CPU Provider cannot capture and replay yet.

## 3. AVX-512

- [x] 3.1 The dense kernels at a second width, AVX-512F, generated from the AVX2 ones. They are used for several rows where the host has AVX-512F; a single row stays on AVX2, since it is bound by memory bandwidth. `MAGNETAR_CPU_SIMD=avx2` keeps AVX2. Test: `both_vector_widths_match_the_reference_cpu`.
- [x] 3.2 Benchmark CPU (run 44529384440, AMD EPYC 9V74 with AVX-512, 4 vCPUs):
  - Magnetar: TTFT 48.2 ms, about 34.3 ms per token in steady state (one of the five requests ran at about 68 ms per token);
  - Ollama: prompt 26.8 ms (4 tokens cached), 25.1 ms per token.

  llama-server did not build in this run.

## 4. Stored orientation, decode kernels and pool size

- [x] 4.1 A compact weight with more rows than columns is held transposed and used the other way round; reads, shapes and lookups (a column gather) return its logical values.
- [x] 4.2 Decode kernels: sixteen outputs (eight on AVX2) at a time output-major, eight weight rows per pass input-major; AVX-512 for one row too.
- [x] 4.3 Splits: row blocks for input-major weights narrower than 4,096 columns per thread; one or two wide column blocks per thread otherwise; output-major tasks of whole groups of sixteen outputs.
- [x] 4.4 Pool size: one thread per physical core was tried and measured slower on the runners (design, thread count). The default stays every logical CPU, and the CPU benchmark reruns Magnetar on one thread per physical core.
- [x] 4.5 Tests:
  - `matmul_kernels_and_lookups_read_compact_weights` over a tall and a wide weight, used transposed and not, with 1 and 3 rows;
  - one-row cases with tails in `dense_matmuls_match_the_reference_cpu` and `both_vector_widths_match_the_reference_cpu`;
  - `pool_and_streamed_matmul_costs` (diagnostic) streams each decode shape of a Qwen2.5-0.5B layer in both layouts.
- [x] 4.6 Local qualification on Qwen2.5-0.5B (4 cores without SMT, AVX-512, profiling on, the same 16 generated tokens): TPOT median 43.3 ms before, 35.4 ms after; `matmul-residual` 0.310 to 0.179 ms per call, `matmul-bias` 0.060 to 0.050 ms.
- [x] 4.7 Benchmark CPU, all engines in one job (run 45343428878, AMD EPYC 9V74 with AVX-512, 2 cores × 2 threads, providers/cpu 3c01cf3):

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar `cpu-native`, 2 threads (one per core) | 620 ms | 81.3 ms | 35.8 ms |
  | Magnetar `cpu-native`, 4 threads (every logical CPU) | 595 ms | 71.5 ms | 34.8 ms |
  | llama-server (FP16 GGUF) | 502 ms | — | 29.7 ms |
  | Ollama (FP16 GGUF, 4 prompt tokens cached) | 460 ms | — | 26.8 ms |

  - The 16 generated tokens are those of earlier runs (" Paris. It is the largest city in Europe and the third largest city in the").
  - Run 44529384440 (the same CPU model, before section 4) measured about 34.3 ms per token in steady state. So the long-row kernels gave the runner much less than the local 18%, and its per-token time is about the same as before.
  - The remaining gap to Ollama is about 8 ms per token and 45 ms of TTFT.
- [x] 4.8 TTFT on the runner went from 48.2 ms (run 44529384440) to 71.5 ms (run 45343428878, every logical CPU), on the same CPU model (EPYC 9V74), while TPOT stayed about 35 ms. Locally (4 cores without SMT), on the same machine, nothing regressed:

  | Magnetar | CPU Provider | TTFT | TPOT |
  | --- | --- | --- | --- |
  | 522d2e4 (before the merge with main) | 8324c36 | 94.7 ms | 48-50 ms |
  | 2e5e7f0 (after the merge) | 20b34ce | 95 ms | 48.2 ms |
  | dab7c3f (section 4) | 3c01cf3 | 87.3 ms | 38.3 ms |

  Step capture still replays: with `MAGNETAR_CPU_STEP_CAPTURE=0` the last row becomes TTFT 110.6 ms and TPOT 49.8 ms. The runner's regression may be specific to Zen 4, in the several-row (prefill) kernels that section 4's orientation routes the down, k and v projections through. Section 5 replaces those kernels. The next campaign tells whether the runner's TTFT comes back.

## 5. Register-tiled prefill kernels

- [x] 5.1 Several rows of `a` run register tiles of up to six rows by four vectors (AVX-512) or two (AVX2), accumulated across the whole reduction. Each weight vector is loaded once for every row of the tile, and outputs are stored once. Input-major tiles walk their weight rows in blocks of 32 (unblocked, a wide weight read one page per load). The tests cover 3, 5 and 8 rows with tails (`dense_matmuls_match_the_reference_cpu`, `both_vector_widths_match_the_reference_cpu`).
- [x] 5.2 Locally, 5 rows, AVX-512:
  - gate/up 0.634 to 0.268 ms;
  - the down projection 0.598 to 0.26-0.33 ms;
  - Qwen2.5-0.5B TTFT 87.3 to 71.8-78.4 ms, with the same 16 tokens (providers/cpu e135ad2).
- [x] 5.3 Benchmark CPU, all engines in one job (run 45353462271, Intel Xeon Platinum 8370C with AVX-512, 2 cores × 2 threads, providers/cpu e135ad2). Magnetar's 16 tokens are llama-server's:

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar `cpu-native`, 4 threads (default) | 632 ms | 61.5 ms | 38.1 ms |
  | Magnetar `cpu-native`, 2 threads (one per core) | — | 59.6 ms | 39.3 ms |
  | llama-server (FP16 GGUF) | 749 ms | — | 46.3 ms |
  | Ollama (FP16 GGUF, `n_threads = 2`) | 763 ms | — | 44.2 ms |

  On this host Magnetar is ahead of both, per request and per token. The host is an Intel CPU, not the Zen 4 of runs 44529384440 and 45343428878, so the Zen 4 TTFT question of task 4.8 needs a campaign on a Zen 4 runner.

## 6. Gated matmuls and kernel diagnostics on the runners

- [x] 6.1 Both CPU Providers advertise `gated-matmul` with the Runtime's Reference semantics, so the Runtime folds a gated MLP's two matmuls and `silu-mul` into one node. The optimized CPU Provider computes both products as the matmul family does (packed, compact or f32) and applies the activation in place. Test: `gated_matmuls_match_the_reference_cpu` (compact, f32 and packed weights; 1 and 5 rows). Locally, over 64 tokens: the same text, 0.602 ms per layer against 0.614 ms for the two matmuls and `silu-mul` (providers/cpu 6ec6040).
- [x] 6.2 A real 64-token profile puts the decode step at its kernels' sum: 34.9 ms against a TPOT of 37.5 ms. The effective bandwidths are about 29 GB/s for gate/up, 30 GB/s for o and down, 40 GB/s for the LM head (6.9 ms per token) and 14 GB/s for the small q, k and v products. The isolated streaming diagnostic reads 400 MB per shape, part of which stays in this host's 260 MB L3, so it overstates those figures.
- [ ] 6.3 The CPU benchmark streams each decode shape on the runner, with the default kernels and with AVX2 only (step "CPU kernel bandwidth"). The kernel choices (rows per pass, outputs per tile, vector width) are then checked on each runner CPU model, Zen 4 first.
