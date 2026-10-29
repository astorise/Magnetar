## 1. Scope and compatibility

- [ ] 1.1 Reconcile E03, E04 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. E03 Asynchronous CPU GPU scheduling

- [ ] 2.1 Implement E03: Runtime scheduling SHALL overlap preparation of batch N+1 with forward execution of N using immutable metadata generations, explicit dependencies and completion barriers.
- [ ] 2.2 Add the positive conformance case: when independent schedule preparation overlaps device work, verify trace evidence demonstrates overlap with correct output attribution and bounded buffers.
- [ ] 2.3 Add the failure/isolation case: when cancellation, metadata mutation or a failed completion races with overlap, verify dependent submissions are withheld and resources remain protected until failure resolution.

## 3. E04 Batch class graph replay

- [ ] 3.1 Implement E04: Serving SHALL bind CUDA replay to admitted batch classes with bounded padding, valid KV/weight leases, observable cache behavior and an uncaptured fallback.
- [ ] 3.2 Add the positive conformance case: when a request batch fits a qualified replay class, verify padding, useful token work, capture memory and replay benefit are measured.
- [ ] 3.3 Add the failure/isolation case: when the batch changes shape, loses a lease or exceeds a class budget, verify the class is invalidated or uncaptured execution is selected without stale binding reuse.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
