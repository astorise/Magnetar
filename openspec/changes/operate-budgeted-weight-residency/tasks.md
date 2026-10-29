## 1. Scope and compatibility

- [ ] 1.1 Reconcile D02, D03, D04, D05, D11 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. D02 Explicit weight residency planning

- [ ] 2.1 Implement D02: WeightResidencyPlan SHALL extend existing MemoryManager pools and ResourceAffinity with bounded resident, pinned, streamed, prefetch and evict decisions, generation identities and explicit movement.
- [ ] 2.2 Add the positive conformance case: when a model plan assigns weights to SSD, RAM and VRAM tiers, verify each replica or transfer has an accounted lease and affinity-compatible consumer.
- [ ] 2.3 Add the failure/isolation case: when an eviction targets an in-flight or pinned weight, verify the transition waits or is rejected without invalidating active execution.

## 3. D03 Measured cost aware residency

- [ ] 3.1 Implement D03: Residency policy SHALL use size, observed frequency, predicted next use, I/O and conversion costs and memory pressure with bounded decisions and auditable evidence.
- [ ] 3.2 Add the positive conformance case: when competing residency choices fit the admitted budget, verify the selected policy is compared against fixed-budget alternatives using transfer and stall cost.
- [ ] 3.3 Add the failure/isolation case: when predictions are missing or oscillate, verify a conservative bounded policy and hysteresis prevent unbounded thrashing.

## 4. D04 Authorized bounded weight streaming

- [ ] 4.1 Implement D04: Streaming SHALL read only authorized validated local artifact ranges through bounded host/device buffers and asynchronous prefetch with explicitly supported Provider movement.
- [ ] 4.2 Add the positive conformance case: when a model larger than Device memory executes through a supported streaming profile, verify peak memory respects admission and I/O bytes, stalls and useful prefetch are measured.
- [ ] 4.3 Add the failure/isolation case: when a source read fails, bytes change or the Provider lacks the required operation, verify the affected generation is rejected or safely aborted without publishing unverified weights.

## 5. D05 Completion safe staged buffers

- [ ] 5.1 Implement D05: Double/triple buffering SHALL enforce EMPTY to INFLIGHT to READY to LEASED to RECLAIM transitions and publish READY only after successful I/O, validation and Device synchronization.
- [ ] 5.2 Add the positive conformance case: when a prefetched buffer finishes all required operations, verify consumers acquire the correct generation and reclamation waits for CompletionToken fences.
- [ ] 5.3 Add the failure/isolation case: when I/O fails or cancellation races with an in-flight copy, verify the buffer never becomes READY and is quarantined or reclaimed only after completion is safe.

## 6. D11 Pressure driven memory admission

- [ ] 6.1 Implement D11: Memory control SHALL enforce separate and total quotas for weights, KV, workspace, prefetch and captures, using gradual completion-safe evictions and explicit admission/rejection.
- [ ] 6.2 Add the positive conformance case: when memory pressure rises during concurrent generation, verify admission and evictions respect reservations and report the exhausted class.
- [ ] 6.3 Add the failure/isolation case: when no compatible eviction or transfer can meet the budget, verify new work is rejected or queued by policy without implicit migration or stealing active leases.

## 7. Qualification, rollout and documentation

- [ ] 7.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 7.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 7.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 7.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 7.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 7.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 7.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
