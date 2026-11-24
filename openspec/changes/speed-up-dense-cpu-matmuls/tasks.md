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

  Step capture still replays: with `MAGNETAR_CPU_STEP_CAPTURE=0` the last row becomes TTFT 110.6 ms and TPOT 49.8 ms. The runner's regression may be specific to Zen 4, in the several-row (prefill) kernels that section 4's orientation routes the down, k and v projections through. Section 5 replaces those kernels.

  Resolved: on the same CPU model with the section 5 and 6 kernels (run 45387558163, EPYC 9V74, providers/cpu 03706fb), the five measured requests' TTFT are 46.3-47.4 ms, back below the 48.2 ms of run 44529384440.

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

  A second Intel host (run 45382158317, Intel Xeon 6975P-C, 2 cores × 2 threads, same commit, the same tokens as llama-server):

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar `cpu-native`, 4 threads (default) | 618 ms | 80-83 ms | 35.3-35.8 ms |
  | Magnetar `cpu-native`, 2 threads (one per core) | — | 59-60 ms | 38-39 ms |
  | llama-server (FP16 GGUF) | 755 ms | — | 45.3 ms |

  Ollama did not run: the benchmark's Ollama adapter refused the placement Ollama reported (`size_vram=987926364` on a CPU-only job). Since section 5, the prefill is compute-bound: two hardware threads on one core compete for its FMA units, and on this host one thread per core prefills about 22 ms faster, while the decode, bound by bandwidth, is 3 ms per token faster on every logical CPU. Whether the prefill should use one thread per core is decided on more runner models (task 6.3).

## 6. Gated matmuls and kernel diagnostics on the runners

- [x] 6.1 Both CPU Providers advertise `gated-matmul` with the Runtime's Reference semantics, so the Runtime folds a gated MLP's two matmuls and `silu-mul` into one node. The optimized CPU Provider computes both products as the matmul family does (packed, compact or f32) and applies the activation in place. Test: `gated_matmuls_match_the_reference_cpu` (compact, f32 and packed weights; 1 and 5 rows). Locally, over 64 tokens: the same text, 0.602 ms per layer against 0.614 ms for the two matmuls and `silu-mul` (providers/cpu 6ec6040).
- [x] 6.2 A real 64-token profile puts the decode step at its kernels' sum: 34.9 ms against a TPOT of 37.5 ms. The effective bandwidths are about 29 GB/s for gate/up, 30 GB/s for o and down, 40 GB/s for the LM head (6.9 ms per token) and 14 GB/s for the small q, k and v products. The isolated streaming diagnostic reads 400 MB per shape, part of which stays in this host's 260 MB L3, so it overstates those figures.
- [x] 6.4 Prefill tiles in blocks of 8 weight rows. A real 2-token profile split by row count put a replayed 5-row prefill's `gated-matmul` at 1.08 ms per layer, against 0.67 ms for one row. The isolated diagnostic had hidden it, since it reused one weight held in cache. Streamed from memory (`MAGNETAR_CPU_BENCH_ROWS` sets the streamed diagnostic's rows), a 5-row gate projection took 0.45 ms with blocks of 32 rows and 0.36 ms with 8. Locally, TTFT went from 66.3 to 62.6-63.9 ms, with the same tokens (providers/cpu 03706fb).
- [ ] 6.3 The CPU benchmark streams each decode shape on the runner, with the default kernels and with AVX2 only (step "CPU kernel bandwidth"). The kernel choices (rows per pass, outputs per tile, vector width) are then checked on each runner CPU model, Zen 4 first. First sample: run 45383670846, Intel Xeon 6973P-C, 2 cores × 2 threads, providers/cpu 6ec6040.
  - In the stored orientations, input-major gate/up streams at 33.0 GB/s and output-major down at 31.7 GB/s. The other layout of each reaches 23 GB/s.
  - AVX2 and AVX-512 are equal within noise on every shape, so a one-row step is bound by memory on this host.
  - All engines in one job, the same tokens:
    - Magnetar: 637 ms per request; TTFT 80.2 ms and 36.8 ms per token on 4 threads, 60.7 ms and 40.7 ms on one thread per core;
    - llama-server: 790 ms per request, 47.7 ms per token.
  - Ollama ran on the CPU (its log: `library=cpu`, no usable GPU) but reported its whole model in `size_vram`, which the benchmark adapter reads as GPU placement. Its cell failed, as on run 45382158317. The adapter now accepts that on the CPU cell only when the server discovered the CPU alone (`--server-log`).

  Second sample, Zen 4: run 45387558163, AMD EPYC 9V74, 2 cores × 2 threads, providers/cpu 03706fb.
  - The stored orientations stream at 36.4 GB/s (gate/up) and 36.1 GB/s (down), against 29-30 GB/s for the other layouts. AVX2 equals AVX-512 again.
  - No kernel choice needs to depend on the CPU model so far: on both hosts the one-row step is bound by memory, and the stored orientations are the faster ones.
  - All engines in one job, the same tokens:

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar `cpu-native`, 4 threads (default) | 557 ms | 46.3-47.4 ms | 34.0 ms |
  | Magnetar `cpu-native`, 2 threads (one per core) | — | 48.3 ms | 34.6 ms |
  | llama-server (FP16 GGUF) | 559 ms | — | 33.9 ms |
  | Ollama (FP16 GGUF, 4 prompt tokens cached, `n_threads = 2`) | 503 ms | — | 29.3 ms |

  Magnetar matches llama-server per request on this host. Ollama's decode is about 4.7 ms per token faster: its kernels reach about 34 GB/s on the model, against Magnetar's 29.
