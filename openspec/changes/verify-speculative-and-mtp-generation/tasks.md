## 1. Scope and compatibility

- [ ] 1.1 Reconcile E08, E09, F03 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. E08 Exact speculative decoding

- [ ] 2.1 Implement E08: Speculative generation SHALL implement draft/verify with target-distribution-preserving acceptance, rejection correction, isolated provisional KV and an ordinary target fallback; n-gram lookup MAY replace a costlier draft.
- [ ] 2.2 Add the positive conformance case: when draft candidates are verified by the target model, verify greedy results match target-only execution and stochastic tests validate the target distribution.
- [ ] 2.3 Add the failure/isolation case: when drafts are rejected, unavailable or too costly at current concurrency, verify unaccepted KV is discarded and target-only generation resumes without committing draft tokens.

## 3. E09 Adaptive compatible MTP verification

- [ ] 3.1 Implement E09: Adaptive verification length SHALL use only declared compatible MTP/draft profiles, bounded confidence/concurrency decisions and the exact verification semantics.
- [ ] 3.2 Add the positive conformance case: when a compatible model supplies MTP candidates under changing load, verify verification length changes within budget and net accepted-token throughput and quality are measured.
- [ ] 3.3 Add the failure/isolation case: when the head contract is absent or speculative overhead exceeds benefit, verify MTP is disabled and exact target-only generation remains available.

## 4. F03 Explicit draft and MTP model semantics

- [ ] 4.1 Implement F03: Model Components SHALL declare supported draft/MTP heads, candidate-token semantics, required state and verification compatibility without owning token acceptance or selecting Devices.
- [ ] 4.2 Add the positive conformance case: when a compatible Component supplies candidates to generic speculation, verify Runtime target verification owns commits and the Component receives only inference-scoped portable resources.
- [ ] 4.3 Add the failure/isolation case: when a model has no compatible draft/MTP head, verify the extension is Unsupported without preventing ordinary dense or MoE execution.

## 5. Qualification, rollout and documentation

- [ ] 5.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 5.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 5.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 5.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 5.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 5.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 5.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
