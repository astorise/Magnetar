## 1. Scope and compatibility

- [ ] 1.1 Reconcile D08, D09 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. D08 Qualified quantized KV storage

- [ ] 2.1 Implement D08: Quantized INT8, FP8 or Q4 KV SHALL be enabled only with compatible attention kernels, explicit scale/layout metadata and context-length quality qualification.
- [ ] 2.2 Add the positive conformance case: when a supported long-context generation uses quantized pages, verify memory savings and quality/latency are measured across declared context lengths.
- [ ] 2.3 Add the failure/isolation case: when the attention implementation cannot consume the selected KV format, verify planning rejects the profile or performs an explicitly authorized conversion before use.

## 3. D09 Explicit KV tier movement

- [ ] 3.1 Implement D09: KV offload/resume across GPU, RAM and SSD SHALL use bounded, completion-aware transfers with declared costs, integrity checks and optional measured prefetch.
- [ ] 3.2 Add the positive conformance case: when a session resumes from an admitted lower-tier KV copy, verify fresh Device leases are bound and transfer cost is compared with re-prefill.
- [ ] 3.3 Add the failure/isolation case: when a page is corrupted, unavailable or still being written, verify restore fails atomically without reusing stale KV and an authorized re-prefill or error is reported.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
