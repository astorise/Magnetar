## 1. Implementation

- [x] 1.1 New test in `integration-tests/multi-device-cpu-cuda`: a real `MultiDevicePlacementPlan` built from two real GPUs' own real Device metadata, reaching `Ready` via `add_binding` (no stages/movement edges needed for this narrower test).
- [x] 1.2 Real `Ready` -> `Invalidated` transition via the existing, unmodified `transition_to`; verified `accepts_new_work()` becomes `false`.
- [x] 1.3 Verified the real state machine rejects `Invalidated` -> `Ready` with `MultiDevicePlacementError::InvalidStateTransition`, and that the rejected attempt leaves `plan.state` unchanged.
- [x] 1.4 Honestly documented, in the test's own doc comment, that this is not a hardware-failure-injection test -- the "loss" event is caller-driven; only the Plan's source data and the state machine are real.
- [x] 1.5 Gracefully skips on any host with fewer than two real CUDA devices.
- [x] 1.6 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean.

## 2. Verification

- [x] 2.1 Real CI green (`quality.yml`) on GitHub's GPU-less runners, confirming graceful skip.
- [x] 2.2 Verified genuinely executing (not skipped) on the real two-GPU `arc-gpu-magnetar` CI node via `gpu-runner-smoke.yml` (`35012051941`).

## 3. Documentation

- [x] 3.1 `openspec validate add-real-device-loss-degraded-replan-state-machine --strict` passes.
- [x] 3.2 README.md's top-level scope-charter status updated once archived, closing out the four-item follow-up from `add-real-second-gpu-cuda-provider`.
