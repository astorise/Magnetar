## 1. Scope and compatibility

- [ ] 1.1 Reconcile C06, C08 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. C06 Bounded off path autotuning

- [ ] 2.1 Implement C06: Autotuning SHALL run outside token dispatch on already-qualified candidates with explicit compute/memory/time budgets, canaries, hysteresis and rollback at safe generation boundaries.
- [ ] 2.2 Add the positive conformance case: when a tuning campaign yields a qualified improvement, verify a new generation is promoted after canary checks while in-flight work retains its original leases.
- [ ] 2.3 Add the failure/isolation case: when canary tail latency regresses or campaign budgets are exhausted, verify promotion stops and the last valid generation remains available without blocking decode.

## 3. C08 External optimization campaign boundary

- [ ] 3.1 Implement C08: The external Optimization Plane SHALL produce benchmark evidence, signed artifacts and recommendations; Runtime SHALL verify admission and control promotion without granting the plane inference resource or decode-loop compilation authority.
- [ ] 3.2 Add the positive conformance case: when a signed campaign result is proposed, verify Runtime verifies provenance, qualification and current compatibility before accepting a new generation.
- [ ] 3.3 Add the failure/isolation case: when a recommendation is unsigned, stale, revoked or requests execution privileges, verify Runtime rejects it and continues the qualified pinned generation.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
