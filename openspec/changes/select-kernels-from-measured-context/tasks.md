## 1. Scope and compatibility

- [ ] 1.1 Reconcile C01, C05, C09 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

- [ ] 1.4 Concrete contract migration: reconcile `multi-device-placement` / "Kernel Registry Candidate Selection Is Provider-Ranked, Not Provider-Filtered By Affinity" with "Kernel Candidate Enumeration Is Distinct From Affinity Safe Selection" and test both old baseline and new qualified profile.

## 2. C01 Measured benchmark to selection pipeline

- [ ] 2.1 Implement C01: Runtime selection SHALL consume qualified real KernelBenchmark evidence through KernelPerformanceModel and KernelSelectionPolicy only after compatibility, trust, affinity and memory eligibility.
- [ ] 2.2 Add the positive conformance case: when two eligible kernel candidates have versioned measurements, verify selection records the evidence and selects under the declared objective.
- [ ] 2.3 Add the failure/isolation case: when a faster candidate has invalid trust, incompatible affinity or an infeasible workspace, verify it is excluded before ranking regardless of benchmark score.

## 3. C05 Context specialized kernel selection

- [ ] 3.1 Implement C05: Selection SHALL specialize eligible candidates by prefill/decode, batch size, context length, memory pressure and hardware architecture using versioned evidence.
- [ ] 3.2 Add the positive conformance case: when the workload changes from prefill to a qualified decode class, verify selection uses the appropriate evidence without changing graph or model semantics.
- [ ] 3.3 Add the failure/isolation case: when the context is outside measured support or a preferred candidate exceeds budget, verify selection uses a qualified conservative candidate and marks extrapolated evidence explicitly.

## 4. C09 Pinned execution and rollback controls

- [ ] 4.1 Implement C09: Operators SHALL be able to pin kernel/plan generations, disable individual optimizations and roll back at a safe boundary with diagnostic reasons and reproducible evidence.
- [ ] 4.2 Add the positive conformance case: when pinned mode runs an identical qualified profile, verify adaptive retuning and noncontractual nondeterminism are disabled and selected identities are reported.
- [ ] 4.3 Add the failure/isolation case: when a selected generation is revoked or fails at runtime, verify new work stops using it and rollback revalidates trust and resources rather than blindly reviving invalid code.

## 5. Qualification, rollout and documentation

- [ ] 5.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 5.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 5.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 5.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 5.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 5.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 5.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
