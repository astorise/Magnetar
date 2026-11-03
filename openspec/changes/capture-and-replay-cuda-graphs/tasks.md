## 1. Scope and compatibility

- [x] 1.1 Reconcile C04 with the existing contracts linked in proposal.md and record the exact initial supported profile. Recorded in design.md (initial supported profile).
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [x] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation. No WIT change: the contract is Rust-only on `ProviderExecutionApi`.

## 2. C04 Bounded CUDA graph replay

- [x] 2.1 Implement C04: CUDA capture/replay SHALL remain Provider-owned, require admissible shapes and resource lifetimes, reserve capture memory and invalidate replay on resource, stream, version or Device changes.
- [x] 2.2 Add the positive conformance case: when a supported graph is captured and replayed, verify launch/synchronization reduction is measured with preparation and retained memory costs. `a_captured_step_replays_with_an_advanced_position_on_real_hardware`; measured in design.md's evidence.
- [x] 2.3 Add the failure/isolation case: when a captured address lease expires or replay preparation fails, verify replay is invalidated and execution uses a qualified uncaptured path without stale-address access. Invalidation on a replaced input (CUDA) and on an invalidated replay or failed capture (Runtime tests), each falling back to the per-node path.

## 3. Qualification, rollout and documentation

- [x] 3.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 3.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 3.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [x] 3.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed. Run 41652697101, with the per-node profile as the disable control in the same run.
- [x] 3.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change. No native library or generated artifact changes.
- [x] 3.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements. `ProductionLoadedModel::set_step_replay`, per-node fallback, documented limits.
- [ ] 3.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
