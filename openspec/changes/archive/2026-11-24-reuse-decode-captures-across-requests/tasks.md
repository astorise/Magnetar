## 1. Runtime

- [x] 1.1 `PooledDecodeKey`, `PooledDecodeRecordings` and their `StepReplay` methods; `RecordingStage`, which the bucketed prefill's sequence now shares.
- [x] 1.2 `execute_pooled_decode_step`:
  - the Warm / Capture / Replay sequence in the set's decode namespace;
  - capture again before the recorded position;
  - decline when the Provider cannot capture or a capture fails.
- [x] 1.3 `release_pooled_kv_sets` releases the set's decode recordings and namespace.
- [x] 1.4 Tests (`first_native_runtime::prefill_buckets::tests`):
  - `pooled_decode_steps_are_recorded_once_for_their_store_set`;
  - `a_shorter_prompt_captures_its_store_set_decode_step_again`;
  - `a_failed_pooled_decode_capture_falls_back_to_the_request`;
  - `dropping_a_store_set_releases_its_decode_recording_and_namespace`.

## 2. Qualification

- [x] 2.1 Benchmark GPU on Qwen2.5-0.5B, all engines in one job (run 44516315731, RTX 3060). Every measured request's inter-token latencies are 4.1 ms, with no slow first decode step, and the tokens are the same:

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar (CUDA) | 72.9 ms | 10.7 ms | 4.1 ms |
  | Ollama (FP16 GGUF) | 80.0 ms | 5.0 ms | 4.54 ms |

  Before this change, run 44490170115 measured Magnetar at 92.6 ms per request, with its first two decode steps at 11.7-16.7 ms. llama-server did not build in this run (llama.cpp's prebuilt web UI assets were missing; `benchmark/ci.sh` now builds it without them).

  With llama-server, all engines in one job (run 45346099731, RTX 3060, driver 595.71.05, Magnetar aa35dbf). Magnetar's 16 tokens are llama-server's:

  | Engine | Request | TTFT / prompt | Per token |
  | --- | --- | --- | --- |
  | Magnetar (CUDA) | 74 ms | 10.7 ms | 4.16 ms |
  | llama-server (FP16 GGUF) | 80 ms | — | 4.87 ms |
  | Ollama (FP16 GGUF) | 80 ms | — | 4.56 ms |
