## 1. Scope and compatibility

- [ ] 1.1 Reconcile B06, B07 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. B06 Qualified fused kernels

- [ ] 2.1 Implement B06: Providers SHALL qualify RMSNorm plus residual, RoPE plus QKV, SiLU/SwiGLU plus MatMul and epilogue fusion against the unfused graph with equivalent ordering, aliasing and dtype semantics.
- [ ] 2.2 Add the positive conformance case: when a validated fusion pattern is selected, verify fused outputs meet numerical tolerances and launches, synchronization and workspace are compared with the unfused path.
- [ ] 2.3 Add the failure/isolation case: when an alias, layout or shape violates the fusion preconditions, verify the unfused qualified path is selected and no partial graph rewrite is published.

## 3. B07 Optimized attention profiles

- [ ] 3.1 Implement B07: Providers SHALL implement explicitly qualified FlashAttention-style or SDPA attention profiles covering GQA, masks, long contexts, prefill and incremental decode with declared workspace and dtype limits.
- [ ] 3.2 Add the positive conformance case: when supported GQA and masking cases execute through a real checkpoint, verify logits, KV growth, memory and phase latency match the profile acceptance criteria.
- [ ] 3.3 Add the failure/isolation case: when a mask, context length or head arrangement is unsupported, verify planning rejects that optimized candidate and retains a valid reference attention path.

## 4. Qualification, rollout and documentation

- [ ] 4.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 4.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 4.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 4.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 4.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 4.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 4.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
