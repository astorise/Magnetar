## 1. Scope and compatibility

- [ ] 1.1 Reconcile D10, G07 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. D10 Versioned inference state snapshot

- [ ] 2.1 Implement D10: Local serialization SHALL capture a consistent committed token boundary and model/component/tokenizer/adapter identities, KV or recurrent state, RNG, positions, dtype/layout, version and checksums without native handles.
- [ ] 2.2 Add the positive conformance case: when a compatible snapshot is restored into newly admitted resources, verify subsequent generation follows its declared numerical and RNG continuity profile.
- [ ] 2.3 Add the failure/isolation case: when the snapshot is truncated, incompatible, untrusted or exceeds destination quotas, verify restore rejects before publishing session state and releases only safely completed allocations.

## 3. G07 Authorized portable state export import

- [ ] 3.1 Implement G07: InferenceStateExport/Import SHALL define a versioned bounded envelope with model digests, dtype/layout, positions, component/adapter/tokenizer and state identities, integrity and explicit source/destination authorization; native handles SHALL NOT cross nodes.
- [ ] 3.2 Add the positive conformance case: when an authorized destination imports a compatible snapshot, verify it validates trust and capacity and binds fresh local resources before atomically exposing the resumed session.
- [ ] 3.3 Add the failure/isolation case: when the envelope is replayed outside its authorization, corrupted or incompatible, verify import rejects it without state publication and reports a redacted compatibility reason.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
