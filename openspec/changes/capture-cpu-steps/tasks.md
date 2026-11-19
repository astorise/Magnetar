## 1. Optimized CPU Provider

- [x] 1.1 `native::capture`: recordings, advancing offsets, and failures.
- [x] 1.2 `begin_step_capture`, `end_step_capture`, `abort_step_capture`, `replay_step`, `release_step_capture`, `select_greedy_column`, and capture failure on host writes.
- [x] 1.3 Tests: `captured_steps_replay_with_advanced_attributes` and `greedy_columns_are_selected_in_place`. Production-loading's tiny-checkpoint logit comparisons on both CPU Providers now run through capture (47 passed).

## 2. Qualification

- [x] 2.1 Local, Qwen2.5-0.5B, qualification campaign (16 tokens, 5 runs, 4 threads): identical tokens with capture on and off. The median time per output token is 68 ms with capture against 84 ms without; the container's timings are noisy.
- [ ] 2.2 Benchmark CPU, all engines in one job.
