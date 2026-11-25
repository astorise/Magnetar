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
- [x] 6.3 The CPU benchmark streams each decode shape on the runner, with the default kernels and with AVX2 only (step "CPU kernel bandwidth"). The kernel choices (rows per pass, outputs per tile, vector width) are then checked on each runner CPU model, Zen 4 first. First sample: run 45383670846, Intel Xeon 6973P-C, 2 cores × 2 threads, providers/cpu 6ec6040.
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

  Third sample, Intel with the 8-row prefill blocks: run 45387956437, Intel Xeon 6975P-C (AMX), 2 cores × 2 threads, Magnetar 8add7ce, providers/cpu 03706fb.
  - The stored orientations stream at 34.8 GB/s (gate/up) and 33.4 GB/s (down), against 24-25 GB/s for the other layouts. AVX2 equals AVX-512 again.
  - The Ollama cell passed: its placement evidence records the CPU as the only discovered compute, with the repacked weights in `size_vram`.
  - All engines in one job, 5 prompt and 16 generated tokens. Magnetar's bf16 weights pick "third" at token 12, where both FP16 GGUF engines pick "second"; the 11 tokens before match.

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar `cpu-native`, 4 threads (default) | 565 ms | 49.6 ms (48.1-84.9) | 34.3 ms |
  | Magnetar `cpu-native`, 2 threads (one per core) | — | 57.0 ms | 37.5 ms |
  | llama-server (FP16 GGUF) | 741 ms | — | 44.6 ms |
  | Ollama (FP16 GGUF, 4 prompt tokens cached, `n_threads = 2`) | 738 ms | — | 42.8 ms |

  On Intel, the 8-row blocks moved the prefill on 4 threads from 80.2 ms (run 45383670846, Xeon 6973P-C) to 49.6 ms, now ahead of one thread per core. Magnetar leads llama-server and Ollama by about 24% per request on this host. The Xeon 8370C runner has no kernel diagnostic sample yet.

  Conclusion, closed on the samples the hosted pool has drawn since the diagnostic exists: Intel Xeon 6973P-C and 6975P-C, AMD EPYC 9V74 (Zen 4) and 9V45 (Zen 5), Intel Xeon Platinum 8573C.
  - The stored orientations were the faster ones, and AVX2 equal to AVX-512, on the first three samples above (6973P-C, 9V74, 6975P-C). The later samples recorded the stored orientations' bandwidth only: 30.1 and 29.3 GB/s on the 8573C, 38-42 GB/s on every shape on the 9V45.
  - No kernel choice (rows per pass, outputs per tile, vector width) depends on the CPU model. What differs between models is the bandwidth the kernels reach (29-42 GB/s), which task 7.10 follows.
  - The Xeon 8370C was drawn only before the diagnostic step existed (run 45353462271) and has no sample. The hosted pool does not let a run choose its CPU model, so this stays unmeasured rather than open; a later sample on it reopens the check.

## 7. Decode step streaming (Ollama's decode on Zen 4)

Ollama decoded about 4.7 ms per token faster on the Zen 4 runner (6.3). Locally, the replayed step ran its dense products at their kernels' streaming speed, but the small ones at half the large ones' bandwidth, with about 6 ms per token outside the products.

- [x] 7.1 Diagnostics: `decode_step_matmul_costs` streams a step's real matmul sequence (24 layers and the LM head, about 1 GB) and compares variants step by step in one process; `MAGNETAR_CPU_PROFILE` now attributes grouped products. The local host's DRAM is shared, and runs of one build varied by up to 8 ms per token, so variants were compared in one process or by interleaved runs of one binary per version.
- [x] 7.2 Software prefetch of the next rows in the one-row kernels (design "Decode step streaming"); AVX-512 and AVX2.
- [x] 7.3 Grouped query, key and value projections at replay, the value projection moved ahead of the key's rope and KV write. Test: `replayed_product_groups_match_products_run_one_by_one` (1 and 3 rows; a product of another input passed over, a product writing the shared input kept after).
- [x] 7.4 `gated-matmul`'s two products in one pool run, then the sums and activation across the pool. Test: `gated_matmuls_match_the_reference_cpu` now also checks bit-for-bit equality with the two products then `silu_times`, for weights held as written and transposed.
- [x] 7.5 Bias and residual sums in place; a block scan for greedy selection, its block maximum in sixteen lanes (264 to 47.5 us for the 151,936 logits; test: `greedy_columns_match_a_one_by_one_scan`); borrowed invocations at replay.
- [x] 7.6 Locally, 8 interleaved runs of a 64-token profile per version: TPOT median 37.6 ms before, 34.3 ms after (paired difference -3.9 ms on average), with the same tokens (providers/cpu 2e523a4).
- [x] 7.7 The CPU benchmark on the runners, Zen 4 first: Magnetar against llama-server and Ollama per token. Measured on five hosts; the result per host is in the table after the fourth sample, and the gap it shows on AMD is task 7.10.

  First sample: run 45406355532, Intel Xeon 6975P-C (AMX), 2 cores × 2 threads, Magnetar d16a71c, providers/cpu 2e523a4, against run 45387956437 on the same CPU model (providers/cpu 03706fb). The same 16 tokens.

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar, 4 threads (default) | 565 → 559 ms | 49.6 → 46.1 ms | 34.3 → 34.2 ms |
  | Magnetar, 2 threads (one per core) | — | 57.0 → 52.5 ms | 37.5 → 32.5 ms |
  | llama-server (FP16 GGUF) | 748 ms | — | 45.0 ms |
  | Ollama (FP16 GGUF, 4 prompt tokens cached) | 741 ms | — | 43.3 ms |

  - Streamed alone, the `[896, 896]` input-major product (query and o projections) went from 24.2 to 31.1 GB/s, the gate/up projection from 34.8 to 36.5 GB/s.
  - On one thread per core the decode step gained 5 ms (13%), and it is now faster there than on every logical CPU (32.5 against 34.2 ms). On every logical CPU it did not change: two hardware threads of a core share its line fill buffers, which the prefetches now keep busy from one thread. The prefill still prefers every logical CPU (46.1 against 52.5 ms).
  Second sample: run 45435917000, Intel Xeon Platinum 8573C, 2 cores × 2 threads, Magnetar cb71986, providers/cpu c2967a3. The same 16 tokens.

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar, 4 threads (default) | 620 ms | 49.9 ms | 37.9 ms |
  | Magnetar, 2 threads (one per core) | — | 56.8 ms | 39.9 ms |
  | Magnetar, decode on 2 threads, prefill on 4 | — | 51.9 ms | 39.8 ms |
  | llama-server (FP16 GGUF) | 720 ms | — | 43.7 ms |
  | Ollama (FP16 GGUF, 4 prompt tokens cached) | 790 ms | — | 46.4 ms |

  - On this host every logical CPU decodes faster (37.9 against 39.8 ms), unlike the Xeon 6975P-C: the better thread count for the decode step depends on the CPU model.
  - Streamed alone, its kernels reach 30.1 GB/s (gate/up) and 29.3 GB/s (down), below the Xeon 6 hosts.

  Third sample: run 45442269420, AMD EPYC 9V45 (Zen 5), 2 cores × 2 threads, Magnetar 17d4ff9, providers/cpu 3b22bd5. The same 16 tokens.

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar, 4 threads (default) | 440 ms | 35.5 ms | 27.0 ms |
  | Magnetar, 2 threads (one per core) | — | 43.6 ms | 26.6 ms |
  | Magnetar, decode on 2 threads, prefill on 4 | — | 35.5 ms | 26.9 ms |
  | llama-server (FP16 GGUF) | 380 ms | — | 22.7 ms |
  | Ollama (FP16 GGUF, 4 prompt tokens cached) | 370 ms | — | 21.4 ms |

  - Against run 44525765520 on the same CPU model (before this change's kernels): Magnetar 493 -> 440 ms per request, TTFT 53.5 -> 35.5 ms, 29.2 -> 27.0 ms per token; llama-server 23.2 and Ollama 21.3 ms per token then.
  - On this AMD host the engines built on llama.cpp stream the same 988 MB per token at 44-46 GB/s, Magnetar at 37 GB/s; streamed alone, Magnetar's kernels reach 38-42 GB/s on every shape. The gap is in the kernels' access pattern on this CPU, not in the step's other work: the next lever on AMD.
  - The thread count barely matters here (26.6 against 27.0 ms per token).

  Fourth sample, the same CPU model: run 45447623233, AMD EPYC 9V45, Magnetar 7ade9ad. The same 16 tokens. Magnetar 460 ms per request, TTFT 37.3 ms, 28.4 ms per token (26.4 on one thread per core; 27.1 decoding on 2 threads with the prefill on 4); llama-server 380 ms and 22.9 ms per token; Ollama 370 ms and 21.8 ms per token.
  - Its kernel step added llama.cpp's row-at-a-time pattern (`row_streamed_matmul_costs`), but the test harness ran both streaming diagnostics at once, sharing the memory bandwidth (their lines interleave): no conclusion. The step now runs them one at a time (`--test-threads=1`).

  Per token, Magnetar's default (every logical CPU) against the other two engines on the same host, the same 16 tokens:

  | Host (run) | Magnetar | llama-server | Ollama | Magnetar against the faster of the two |
  | --- | --- | --- | --- | --- |
  | Xeon 6975P-C (45406355532) | 34.2 ms | 45.0 ms | 43.3 ms | 21% faster |
  | Xeon Platinum 8573C (45435917000) | 37.9 ms | 43.7 ms | 46.4 ms | 13% faster |
  | EPYC 9V45, Zen 5 (45442269420) | 27.0 ms | 22.7 ms | 21.4 ms | 26% slower |
  | EPYC 9V45, Zen 5 (45447623233) | 28.4 ms | 22.9 ms | 21.8 ms | 30% slower |
  | EPYC 9V74, Zen 4 (45387558163, before this section's changes) | 34.0 ms | 33.9 ms | 29.3 ms | 16% slower |

  Magnetar leads on the two Intel hosts and trails on AMD. There is no sample on the Zen 4 host with this section's changes.

- [x] 7.8 A decode pool of its own: `MAGNETAR_CPU_DECODE_THREADS=N` runs one-row Kernels (dense and packed products, gated sums, attention over one query) on N threads, Kernels of several rows on the main pool (test: `one_row_kernels_run_on_the_decode_pool`; providers/cpu c2967a3). The CPU benchmark measures it on one thread per physical core (step "Magnetar decoding on one thread per physical core"). Whether it becomes the default on hosts with two threads per core is decided on a Zen 4 sample. First samples disagree: one thread per core decoded faster on the Xeon 6975P-C (32.5 against 34.2 ms), slower on the Xeon Platinum 8573C (39.8 against 37.9 ms).
  - Decision: the default stays every logical CPU; `MAGNETAR_CPU_DECODE_THREADS` stays an opt-in. Decoding on one thread per core against every logical CPU, per token: Xeon 6975P-C 32.5 against 34.2 ms; Xeon Platinum 8573C 39.8 against 37.9 ms; EPYC 9V45 26.6 against 27.0 ms and 26.4 against 28.4 ms (runs 45442269420 and 45447623233); EPYC 9V74 (Zen 4, before the prefetches and the pool) 34.6 against 34.0 ms. The sign changes from one CPU model to the next, and the prefill prefers every logical CPU on every host, so no default serves all of them.
  - The pool of its own with the prefill on every logical CPU did not settle it: 39.8 ms on the 8573C (every logical CPU 37.9), 26.9 and 27.1 ms on the 9V45 (27.0 and 28.4).
  - A post-prefetch sample on the Zen 4 host would change this only if it showed a gain on every host family; the hosted pool does not let a run choose it.
- [x] 7.9 Small Kernels in a replayed step (rmsnorm, rope, kv-write, attention: about 170 calls per token, 9-11 us each for 896 values, about 1.8 ms per token). Timed in the model, an rmsnorm call split into its input copy (1.6 us), its weight copy (2.0), the arithmetic (2.0) and the output store with its result (2.6); a rope call computed `sin_cos` per head (448 times for a 14-head query row).
  - rope computes a row's rotations once for every head; rmsnorm and rope read their inputs where they are stored; an output keeps its slot's key; a replay builds no `KernelResult` (providers/cpu 3b22bd5). The same values and tokens.
  - Result: rope 10 -> 9 us, rmsnorm unchanged at 9-10 us. The copies' cost moved into cache misses: the input was just written by other cores' pool tasks, the norm weight last read a step ago. What remains is memory latency on the step's critical path, not instructions.
- [ ] 7.10 Decode on AMD: on the EPYC 9V45 (Zen 5) llama-server and Ollama stream the same 988 MB per token at 44-46 GB/s, Magnetar at 37 GB/s, and Magnetar's per-token time is 19-24% above llama-server's and 26-30% above Ollama's (task 7.7's table). On the Zen 4 host Ollama was 4.7 ms faster per token before this section's changes. The goal is Magnetar's per-token time at or below llama-server's on both AMD hosts.
  - Lever to test: llama.cpp's row-at-a-time pattern (`row_streamed_matmul_costs`), compared with the Provider's kernels one diagnostic at a time (`--test-threads=1`, which the fourth sample of 7.7 lacked).
  - Evidence needed: a CPU benchmark run on an AMD host after the change, since the hosted pool does not let a run choose its CPU model. Not dispatched yet.
