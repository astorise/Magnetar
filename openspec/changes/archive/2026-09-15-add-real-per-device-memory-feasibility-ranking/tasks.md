## 1. Implementation

- [x] 1.1 New test in `integration-tests/multi-device-cpu-cuda`: two real `PlacementCandidate`s (real GPU 0 capacity vs. an artificially constrained GPU 1 budget), fed through the existing, unmodified `select_lowest_cost_eligible`.
- [x] 1.2 Verified the feasible real candidate is selected and the constrained one is rejected specifically with `MultiDevicePlacementErrorCode::MemoryInfeasible`.
- [x] 1.3 Verified eligibility precedes ranking: the constrained (infeasible) candidate is given a deliberately lower cost and still loses.
- [x] 1.4 Gracefully skips on any host with fewer than two real CUDA devices.
- [x] 1.5 Updated the module's own top-of-file doc comment.
- [x] 1.6 `cargo test`/`cargo fmt`/`cargo clippy --all-targets -- -D warnings` all clean.

## 2. Verification

- [x] 2.1 Real CI green (`quality.yml`) on GitHub's GPU-less runners, confirming graceful skip.
- [x] 2.2 Verified genuinely executing (not skipped) on the real two-GPU `arc-gpu-magnetar` CI node via `gpu-runner-smoke.yml` (`35003146469`): real GPU 0 capacity used, real rejection of the constrained candidate confirmed.

## 3. Documentation

- [x] 3.1 `openspec validate add-real-per-device-memory-feasibility-ranking --strict` passes.
- [x] 3.2 README.md's top-level scope-charter status updated once archived.
