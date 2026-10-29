## 1. Scope and compatibility

- [ ] 1.1 Reconcile B01, B02 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. B01 End to end native half precision

- [ ] 2.1 Implement B01: Validated profiles SHALL preserve F16/BF16 weights, activations, intermediates and KV across ingestion, graph descriptors, preparation and execution, with explicitly declared F32 accumulation or conversion points.
- [ ] 2.2 Add the positive conformance case: when a qualified half-precision checkpoint generates multiple tokens, verify residency and copy traces demonstrate no mandatory global F32 weight expansion.
- [ ] 2.3 Add the failure/isolation case: when a required operator lacks the selected dtype, verify planning returns Unsupported or selects an explicitly authorized conversion/fallback before execution.

## 3. B02 CUDA Tensor Core matrix execution

- [ ] 3.1 Implement B02: The CUDA Provider SHALL implement qualified FP16/BF16 MatMul/GEMM using supported Tensor Core paths with declared accumulation, alignment and epilogues and a verified Reference CPU fallback.
- [ ] 3.2 Add the positive conformance case: when aligned and tail-shaped matrices execute on supported CUDA hardware, verify results meet dtype tolerances and end-to-end gains and memory are measured.
- [ ] 3.3 Add the failure/isolation case: when the Device SM, alignment or precision is unsupported, verify eligibility rejects that candidate and uses a policy-approved baseline without advertising Tensor Core execution.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
