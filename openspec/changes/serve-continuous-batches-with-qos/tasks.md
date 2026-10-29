## 1. Scope and compatibility

- [ ] 1.1 Reconcile E01, E02, E05, E06 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. E01 Production continuous batching loop

- [ ] 2.1 Implement E01: Runtime SHALL admit real production sessions into mixed decode steps with isolated KV, stop conditions, token commits and stream attribution through existing batching contracts.
- [ ] 2.2 Add the positive conformance case: when requests enter and leave a running batch, verify each result matches its per-session reference and released slots are reused only after completion.
- [ ] 2.3 Add the failure/isolation case: when one session cancels or errors during a submitted step, verify only its future work stops and other sessions retain their correct state and outputs.

## 3. E02 Fair chunked prefill

- [ ] 3.1 Implement E02: The scheduler SHALL split supported long prefills by token/time budgets while preserving absolute positions and fair access for active decode work.
- [ ] 3.2 Add the positive conformance case: when long prompts arrive during decode traffic, verify bounded prefill chunks produce the reference logits and report TTFT/TPOT fairness.
- [ ] 3.3 Add the failure/isolation case: when a chunk would exceed memory or the model does not support segmentation, verify admission defers or explicitly rejects chunking without publishing partial invalid KV.

## 4. E05 Local cost aware request routing

- [ ] 4.1 Implement E05: Runtime SHALL route locally among eligible Model Instances using context length, occupied KV, FLOP quota, pressure and TTFT/TPOT objectives after affinity and admission validation.
- [ ] 4.2 Add the positive conformance case: when several compatible local execution candidates are available, verify routing records its costs and preserves session-affine placement unless movement is explicitly authorized.
- [ ] 4.3 Add the failure/isolation case: when the lowest-cost candidate conflicts with active KV affinity, verify it is excluded rather than causing implicit session migration.

## 5. E06 Serving QoS and bounded streaming

- [ ] 5.1 Implement E06: Serving SHALL enforce profile-defined priority, deadlines, fairness, starvation protection, backpressure, per-session cancellation and request budgets.
- [ ] 5.2 Add the positive conformance case: when sessions at concurrency 1, 4, 16 or 32 run when capacity permits, verify tail latency, starvation and cancellation behavior are measured under the declared policy.
- [ ] 5.3 Add the failure/isolation case: when a slow stream consumer or expired deadline exhausts its budget, verify bounded buffering and policy-driven cancellation protect other sessions and preserve completion safety.

## 6. Qualification, rollout and documentation

- [ ] 6.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 6.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 6.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 6.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 6.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 6.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 6.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
