## 1. Scope and compatibility

- [x] 1.1 Reconcile C04 with the existing contracts linked in proposal.md and record the exact initial supported profile. Recorded in design.md (initial supported profile).
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
  - Decided by the owner on 2026-11-25: design.md "Qualification gates". Closes with the first GPU run of `captured_step_device_memory`, whose report completes the memory budget.
- [x] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation. No WIT change: the contract is Rust-only on `ProviderExecutionApi`.

## 2. C04 Bounded CUDA graph replay

- [x] 2.1 Implement C04: CUDA capture/replay SHALL remain Provider-owned, require admissible shapes and resource lifetimes, reserve capture memory and invalidate replay on resource, stream, version or Device changes.
- [x] 2.2 Add the positive conformance case: when a supported graph is captured and replayed, verify launch/synchronization reduction is measured with preparation and retained memory costs. `a_captured_step_replays_with_an_advanced_position_on_real_hardware`; measured in design.md's evidence.
- [x] 2.3 Add the failure/isolation case: when a captured address lease expires or replay preparation fails, verify replay is invalidated and execution uses a qualified uncaptured path without stale-address access. Invalidation on a replaced input (CUDA) and on an invalidated replay or failed capture (Runtime tests), each falling back to the per-node path.

## 3. Qualification, rollout and documentation

- [x] 3.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [x] 3.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
  - No WIT, Component or Operator change (1.3), so this change has no old/new linking to validate; the capture contract is Rust-only on `ProviderExecutionApi`, and no wasm boundary moves.
  - `quality / test (ubuntu-latest)`, `(windows-latest)` and `(macos-latest)` run the Runtime's capture and replay tests (test Provider: `replayed_decode_steps_match_per_node_steps_bit_for_bit`, `an_invalidated_replay_runs_per_node_and_captures_again`, `a_failed_capture_runs_the_step_per_node_and_stops_capturing`, `bucketed_prefills_are_captured_once_then_replayed`, …) at every pull request; `quality / openspec` runs `openspec validate --all --strict` on the deltas. All passed on Magnetar#231 at 84ab32b.
  - Skipped, not passed: CUDA graphs run only on the Linux x86_64 RTX 3060 runner (GPU CI). Windows CUDA, macOS and ARM have no CUDA hardware in this repository's runners.
- [ ] 3.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
  - Covered by tests: failure (`a_failed_capture_runs_the_step_per_node_and_stops_capturing`, `a_failed_pooled_decode_capture_falls_back_to_the_request`, `a_failed_prefill_capture_declines_its_key`); invalid state reuse (`an_invalidated_replay_runs_per_node_and_captures_again`; on CUDA, a replaced input invalidates the capture); release (`dropping_the_engine_releases_captures_its_requests_never_released`); session isolation (`same_request_id_cannot_reuse_or_discard_another_sessions_kv`: another session's `release_request_state` is refused, and the owner check runs before the capture is released).
  - Added on 2026-11-25, Runtime, passing on the hosted runners' tests (a lifecycle harness runs several requests on one engine and one Provider):
    - cancellation and state reuse: `a_request_released_between_replayed_steps_gives_its_capture_back` (its capture is released with it, the next request captures and replays its own, every step bit for bit the per-node one);
    - tenant and session isolation: `requests_of_two_sessions_replay_their_own_captures` (two sessions' requests interleaved, each replays its own capture; another session's release is refused and leaves the capture in place);
    - memory pressure: `a_capture_refused_for_memory_runs_every_step_per_node` (a Provider that cannot reserve the capture's memory: the same values, one attempt, no capture).
  - Added on 2026-11-25, CUDA, for the GPU runner: `a_replay_in_flight_survives_the_retirement_of_its_resources_on_real_hardware` launches a replay, releases the buffers it reads and reuses their memory before anything waits for it, then checks its result and that the capture now refuses to replay. This task closes with its first GPU run.
- [x] 3.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed. Run 41652697101, with the per-node profile as the disable control in the same run.
- [x] 3.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change. No native library or generated artifact changes.
- [x] 3.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements. `ProductionLoadedModel::set_step_replay`, per-node fallback, documented limits.
- [ ] 3.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
