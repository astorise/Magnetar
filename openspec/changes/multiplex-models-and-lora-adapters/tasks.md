## 1. Scope and compatibility

- [ ] 1.1 Reconcile F06, F08 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. F06 Multiplexed LoRA inference

- [ ] 2.1 Implement F06: LoRA/QLoRA inference SHALL validate base-model and target-module compatibility, use dedicated bounded adapter slots and avoid copying the full base model for each adaptation.
- [ ] 2.2 Add the positive conformance case: when concurrent authorized sessions use distinct compatible adapters, verify outputs match isolated adapted references and KV/prefix keys distinguish adapter identities.
- [ ] 2.3 Add the failure/isolation case: when an adapter changes during active work or an unsupported composition is requested, verify the change is rejected or performed at an explicit safe boundary with incompatible cache invalidation.

## 3. F08 Concurrent model instance management

- [ ] 3.1 Implement F08: Runtime SHALL manage multiple Model Instances with explicit warmup, per-model quotas/reservations and model-bound kernel compatibility and cache identities.
- [ ] 3.2 Add the positive conformance case: when multiple real models share admitted local resources, verify warmup and request latency are measured without cross-model state or weight aliasing.
- [ ] 3.3 Add the failure/isolation case: when one model reloads, exceeds quota or changes generation, verify only dependent plans/caches are invalidated and unrelated admitted models retain valid leases.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
