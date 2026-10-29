## 1. Scope and compatibility

- [ ] 1.1 Reconcile F02, F05, B09 with the existing contracts linked in proposal.md and record the exact initial supported profile.
- [ ] 1.2 Confirm upstream exit-gate dependencies, hardware access, checkpoint authorization, numeric/quality thresholds and memory/SLO budgets.
- [ ] 1.3 Prefer current WIT; if new portable operators/state/results are needed, design a versioned additive extension, migration and unsupported behavior before implementation.

## 2. F02 Complete MoE model execution

- [ ] 2.1 Implement F02: MoE Components SHALL represent router, top-k, expert/shared-expert graphs and lifecycles through portable Capabilities while Runtime owns residency and Providers own execution.
- [ ] 2.2 Add the positive conformance case: when a real MoE checkpoint performs prefill and multi-step decode, verify routing, combination weights, expert utilization and logits match the qualified reference.
- [ ] 2.3 Add the failure/isolation case: when routing produces invalid expert identities or insufficient admitted capacity, verify validation fails or uses an explicitly supported fallback before any invalid expert access.

## 3. F05 Trace driven expert residency

- [ ] 3.1 Implement F05: MoE expert caching SHALL use validated traces, hot-layer pinning and top-k prefetch under explicit admission and completion-safe staged buffers.
- [ ] 3.2 Add the positive conformance case: when a real expert access distribution drives a fixed-budget policy, verify useful hit rate, misprefetch, I/O stalls and imbalance are compared against baseline policies.
- [ ] 3.3 Add the failure/isolation case: when routing shifts abruptly or prefetched experts are unused, verify bounded eviction/hysteresis protects active experts and prevents runaway transfer or memory use.

## 4. B09 Quantized expert kernels

- [ ] 4.1 Implement B09: Providers SHALL qualify grouped quantized expert GEMM and fused dispatch/gather/scatter against the portable MoE routing and accumulation semantics.
- [ ] 4.2 Add the positive conformance case: when a supported top-k expert batch executes, verify outputs match unfused expert references including ordering, shared experts and quantization metadata.
- [ ] 4.3 Add the failure/isolation case: when expert routing is empty, skewed or contains unsupported packed layouts, verify the supported fallback or structured rejection preserves token ownership and never reads another expert buffer.

## 5. Qualification, rollout and documentation

- [ ] 5.1 Run affected cargo fmt/check/clippy/tests and conformance in both relevant workspaces and affected extension repositories; keep baseline F32 tests passing.
- [ ] 5.2 Validate any changed WIT and old/new linking, OpenSpec deltas and native/wasm boundaries; test Linux, Windows/macOS and ARM where applicable, with explicit hardware skips.
- [ ] 5.3 Exercise cancellation, failure, memory pressure, invalid artifact/state reuse and tenant/session isolation; prove lease retirement waits for completion.
- [ ] 5.4 Execute versioned before/after real-checkpoint campaigns on appropriate real hardware using shared V0 evidence, recording tail latency, total memory, I/O, numeric and downstream quality. Do not mark skipped hardware passed.
- [ ] 5.5 Check third-party licenses, SBOM, signatures/provenance and artifact reproducibility where native libraries or generated artifacts change.
- [ ] 5.6 Verify targeted disable, safe fallback/Unsupported, pinned mode and generation rollback; document support limits, public architecture/API migration and measurements.
- [ ] 5.7 Update the roadmap coverage/support matrix with evidence and actual maturity only after implementation gates pass; archive only completed scope.
