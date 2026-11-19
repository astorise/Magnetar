## 1. Optimized CPU Provider

- [x] 1.1 `native::capture`: recordings, advancing offsets, and failures.
- [x] 1.2 `begin_step_capture`, `end_step_capture`, `abort_step_capture`, `replay_step`, `release_step_capture`, `select_greedy_column`, and capture failure on host writes.
- [x] 1.3 Tests: `captured_steps_replay_with_advanced_attributes` and `greedy_columns_are_selected_in_place`. Production-loading's tiny-checkpoint logit comparisons on both CPU Providers now run through capture (47 passed).

- [x] 1.4 While a captured bucketed prefill replays (`padding_rows` advancing), row matmuls of more rows than the padding compute only the leading rows, as the CUDA Provider's packed matmuls do; a one-row product is computed whole. The input-major kernel for several rows widens each weight block once for every row (`replayed_prefills_skip_their_padding_rows`). Local TTFT about 96 ms against 150-230 ms, the same tokens.

## 2. Qualification

- [x] 2.1 Local, Qwen2.5-0.5B, qualification campaign (16 tokens, 5 runs, 4 threads): identical tokens with capture on and off. The median time per output token is 68 ms with capture against 84 ms without; the container's timings are noisy.
- [x] 2.2 Benchmark CPU on Qwen2.5-0.5B, all engines in one job (run 44521749985, AMD EPYC 7763, 4 vCPUs, no AVX-512, providers/cpu 65e9627). Same tokens as the GPU runs:

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar `cpu-native` | 600 ms | 101.8 ms | 33.2 ms (32-34 ms once replayed) |
  | llama-server (FP16 GGUF) | 532 ms | 49.4 ms | 32.6 ms |
  | Ollama (FP16 GGUF, 4 prompt tokens cached) | 500 ms | 30.7 ms | 29.2 ms |

  Run 44517360795 (an EPYC 9V74, without capture) measured Magnetar at 752 ms per request and 43.6 ms per token, against 32.3 ms for llama-server on that host. The gap that remains is the TTFT: the 5-token prompt runs as an 8-row bucket.
- [ ] 2.3 Benchmark CPU, all engines in one job, with the padding skip (providers/cpu 5c8cd07).
