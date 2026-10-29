## 1. Scope and compatibility

- [ ] 1.1 Reconcile A06, D01 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. A06 Redacted phase attribution

- [ ] 2.1 Implement A06: Observability SHALL attribute kernel, plan, copy, I/O and cache costs through correlated phase records, use actual device timing when available and redact prompts, weight bytes, tensor values, secrets and sensitive paths.
- [ ] 2.2 Add the positive conformance case: when a request crosses preparation, I/O, copies and compute, verify the report distinguishes device time, host time, overlap and unavailable counters without double counting.
- [ ] 2.3 Add the failure/isolation case: when a prompt or tensor contains a recognizable sentinel, verify logs, traces and exported metrics omit that sentinel and retain only permitted metadata.

## 3. D01 Anonymized access traces and cache simulation

- [ ] 3.1 Implement D01: Offline tools SHALL evaluate LRU, LFU, SLRU, weighted-cost and offline-optimum cache policies from redacted tensor/expert access traces and budget ladders.
- [ ] 3.2 Add the positive conformance case: when a versioned trace is replayed, verify the simulator reports hit, prefetch, bytes moved, stall costs and an offline-oracle bound under identical budgets.
- [ ] 3.3 Add the failure/isolation case: when a trace lacks complete future access information or contains sensitive identities, verify the oracle is marked unavailable and sensitive content is removed before export.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
