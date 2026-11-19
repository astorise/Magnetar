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
