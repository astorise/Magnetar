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
- [x] 4.4 The pool's default size is one thread per physical core (`native::topology`); `MAGNETAR_CPU_THREADS` overrides it.
- [x] 4.5 Tests:
  - `matmul_kernels_and_lookups_read_compact_weights` over a tall and a wide weight, used transposed and not, with 1 and 3 rows;
  - one-row cases with tails in `dense_matmuls_match_the_reference_cpu` and `both_vector_widths_match_the_reference_cpu`;
  - `cpu_lists_parse_as_linux_writes_them` and `the_default_pool_never_exceeds_the_logical_cpus`;
  - `pool_and_streamed_matmul_costs` (diagnostic) streams each decode shape of a Qwen2.5-0.5B layer in both layouts.
- [x] 4.6 Local qualification on Qwen2.5-0.5B (4 cores without SMT, AVX-512, profiling on, the same 16 generated tokens): TPOT median 43.3 ms before, 35.4 ms after; `matmul-residual` 0.310 to 0.179 ms per call, `matmul-bias` 0.060 to 0.050 ms.
- [ ] 4.7 Benchmark CPU on the same machine as Ollama and llama-server, with the thread-count check on every logical CPU.
