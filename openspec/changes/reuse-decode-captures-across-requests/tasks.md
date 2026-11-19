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

- [ ] 2.1 Benchmark GPU (`magnetar`, 16 tokens, 5 runs) on Qwen2.5-0.5B: no slow decode step after the first requests, request latency against run 44490170115, and the same tokens.
