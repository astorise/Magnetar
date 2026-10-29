## 1. Scope and compatibility

- [ ] 1.1 Reconcile B03, B04 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. B03 Direct GGUF quantized execution

- [ ] 2.1 Implement B03: Providers SHALL compute directly from validated Q8_0 and then Q4_K/Q5_K packed weights, preserving block scales, zero conventions and layouts without mandatory whole-model F32 dequantization.
- [ ] 2.2 Add the positive conformance case: when a real checkpoint in a supported quantized profile generates multiple tokens, verify kernel and residency evidence proves packed-weight computation and quality passes.
- [ ] 2.3 Add the failure/isolation case: when a block is truncated or quantization metadata is incompatible, verify ingestion or preparation rejects it before dispatch and does not silently relabel dequantized execution as direct.

## 3. B04 Direct GPTQ and AWQ matrix execution

- [ ] 3.1 Implement B04: Qualified GPTQ/AWQ profiles SHALL validate packing, group sizes, scales, zeros and calibration provenance and execute direct matrix kernels on real checkpoints.
- [ ] 3.2 Add the positive conformance case: when a declared GPTQ or AWQ checkpoint completes generation, verify logits, downstream quality and memory are compared against its calibrated reference.
- [ ] 3.3 Add the failure/isolation case: when a group layout or calibration profile is unsupported, verify the loader or planner returns an explicit incompatibility and any dequantized fallback is opt-in and separately measured.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
