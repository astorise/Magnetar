## 1. Scope and compatibility

- [ ] 1.1 Reconcile G08, G09, G10 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. G08 Tachyon prefill decode integration proposal

- [ ] 2.1 Implement G08: An optional Tachyon adapter SHALL own inter-node routing, transport, global admission and cost-aware prefill/decode transfer while Magnetar retains local validation/execution and standalone operation.
- [ ] 2.2 Add the positive conformance case: when audited and approved interfaces run on real multiple nodes, verify P/D is compared with colocated execution for KV transfer cost, TTFT, throughput and backpressure.
- [ ] 2.3 Add the failure/isolation case: when transport costs dominate, authorization fails or an interface is unapproved, verify the integration remains an unvalidated proposal or is disabled without coupling Runtime Core to Tachyon.

## 3. G09 Distributed session recovery and drain

- [ ] 3.1 Implement G09: The distributed adapter SHALL coordinate checkpoint/resume, draining, quotas and generation rollback with explicit ownership handoff and fencing of the previous token producer.
- [ ] 3.2 Add the positive conformance case: when a node drains or fails after an acknowledged token boundary, verify a compatible checkpoint resumes under one authorized owner with a declared output replay policy.
- [ ] 3.3 Add the failure/isolation case: when the source is partitioned, transfer is partial or the latest state is uncertain, verify fencing prevents duplicate producers and recovery uses a verified boundary or explicit failure rather than silent token loss.

## 4. G10 Evidence gated remote pooling

- [ ] 4.1 Implement G10: Remote memory pooling, external KV and RDMA/NVLink transport experiments SHALL require real topology evidence, explicit authorization, bounded quotas and a measured advantage over simpler transfer or re-prefill.
- [ ] 4.2 Add the positive conformance case: when a justified infrastructure profile runs a pooling experiment, verify latency, throughput, integrity, isolation and failure behavior are measured on the actual interconnect.
- [ ] 4.3 Add the failure/isolation case: when the link is unavailable, unauthorized or slower than the baseline, verify the feature stays disabled and local execution or an explicitly supported transport remains usable.

## 5. Qualification, rollout and documentation

- [ ] 5.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 5.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 5.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 5.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 5.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 5.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 5.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
