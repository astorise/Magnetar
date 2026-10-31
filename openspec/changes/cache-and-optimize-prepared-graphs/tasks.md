## 1. Scope and compatibility

- [ ] 1.1 Reconcile C02, C03 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. C02 Revalidated prepared plan cache

- [ ] 2.1 Implement C02: Prepared plans SHALL be cached with graph, shape, dtype, layout, model, Provider, Device and version fingerprints, bounded storage and validation at every reuse.
- [ ] 2.2 Add the positive conformance case: when a compatible request reuses a cached plan, verify the plan is reused only after its resources, trust generation and capacity assumptions are checked.
- [ ] 2.3 Add the failure/isolation case: when a model, driver, allocation generation or resource lifetime changes, verify the entry is invalidated and preparation or a validated fallback occurs before submission.

## 3. C03 Validated graph optimization passes

- [ ] 3.1 Implement C03: The graph pipeline SHALL type/layout-check propagation, constant folding, dead computation removal and qualified pattern fusion while preserving outputs, side effects, KV writes and portable Component authority.
- [x] 3.2 Add the positive conformance case: when an optimized graph passes semantic validation, verify it produces equivalent logits and lifecycle effects through the existing PreparedExecutionPlan path.
- [ ] 3.3 Add the failure/isolation case: when a pass removes a required KV write or changes aliasing, verify validation rejects the candidate and publishes no partial optimized graph.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
