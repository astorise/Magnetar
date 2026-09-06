## 1. ResourceAffinity: resolved helper, Resident preservation, production validation (design.md Decision 6)

- [ ] 1.1 Add `resolved_resource_affinity(prepared_plan, node, execution_context) -> ResourceAffinity` to `first_native_runtime.rs`, mirroring `resolved_output_placement`'s lookup exactly.
- [ ] 1.2 `dispatch_reference_cpu_operator_multi` calls it for `NodeInputResource::Fresh` resources and its own new outputs (replacing the hardcoded `ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)`).
- [ ] 1.3 For `NodeInputResource::Resident` resources, look up the resource's existing recorded `ResourceAffinity` (via its `TensorResidency`/Memory Manager record) and aggregate it with the dispatch's own affinity through the existing `resource-affinity` aggregation contract, rejecting on conflict rather than overwriting.
- [ ] 1.4 Add the production-level (not `debug_assert!`) structured validation at the Kernel/Runtime dispatch boundary: after `KernelDispatchPlan` is built, reject with a structured error if the invocation's resolved Provider/Device disagrees with the affinity recorded for its resources.
- [ ] 1.5 Test: a Qwen node whose Prepared Plan binds a non-Reference-CPU Provider/Device produces `ResourceAffinity` matching that binding for a Fresh resource.
- [ ] 1.6 Test: a Resident resource's own recorded affinity survives a later dispatch on the same Provider/Device (aggregated, not replaced); a conflicting Provider/Device on the same resource is rejected with a structured error.
- [ ] 1.7 Test: the new production validation actually rejects a deliberately constructed Provider/Device-vs-affinity mismatch, in a release-mode-equivalent test (not relying on `debug_assert!`).
- [ ] 1.8 Full regression: confirm every existing Reference-CPU-only test is bit-for-bit unaffected.

## 2. RMSNorm Host materialization removed (design.md Decision 4)

- [ ] 2.1 Confirm (re-confirm against current code before editing) that `providers/cpu::rmsnorm`/`providers/cuda::CudaKernels::rmsnorm` both accept a `[cols]`-shaped weight and broadcast internally.
- [ ] 2.2 `dispatch_qwen_rmsnorm`'s signature changes `input: HostTensor, weight: HostTensor` to `input: NodeValue, weight: NodeValue`; delete the manual per-row weight broadcast.
- [ ] 2.3 `dispatch_qwen_graph_node`'s "rmsnorm" arm stops calling `.into_host()` on either input before dispatch.
- [ ] 2.4 Update the two large `#[cfg(test)]` oracle functions for the new signature.
- [ ] 2.5 Test: a MatMul output already Resident under the resolved Provider passes into RMSNorm without an intervening `.into_host()` call.
- [ ] 2.6 Full regression: Reference CPU's own RMSNorm numeric output is unchanged.

## 3. RoPE: explicit `head_count` graph attribute, corrected multi-head semantics (design.md Decisions 1-2)

- [ ] 3.1 `operator.rs`'s `"rope"` `OperatorAttributeSchema` gains `head_count` as `OperatorAttributeRule::optional(OperatorAttributeKind::Integer)`.
- [ ] 3.2 Schema test: `head_count` absent -> accepted; `head_count = 1` -> accepted; `head_count > 1` -> accepted; wrong attribute kind (e.g. Float) -> rejected; a still-unknown attribute name -> still rejected.
- [ ] 3.3 `qwen_model_component.rs`'s `rope_q` node construction adds `head_count = a.attention_head_count`; `rope_k` adds `head_count = a.kv_head_count`.
- [ ] 3.4 `providers/cpu::rope` gains `head_count: u64`. Validation: `head_count >= 1`, `cols % head_count == 0`, `dimension > 0`, `dimension % 2 == 0`, `dimension <= cols / head_count`. Loop one extra level (`for head in 0..head_count`), rotating `[head * head_width, head * head_width + dimension)` per row per head (`head_width = cols / head_count`); output starts as a copy of input so untouched columns (partial RoPE's tail) are preserved, not zeroed.
- [ ] 3.5 `providers/cpu` test: `head_count` absent/`1` reproduces today's exact output (regression). New tests: `head_count > 1` with `dimension == head_width` (full multi-head rotation) against a hand-computed expected result; `head_count > 1` with `dimension < head_width` (partial RoPE) confirming the untouched tail is preserved; a GQA-shaped case (`head_count` for K smaller than for Q, over the same base tensor width assumptions) confirming both are computed independently and correctly.
- [ ] 3.6 `providers/cuda/src/kernels.cu`'s `rope_kernel` gains `head_count`; re-derive `row`/`head`/`pair`/`col_base = head * head_width` from the flattened thread index exactly as design.md's Decision 2 specifies (thread count `rows * head_count * (dimension / 2)`, not `rows * head_count * (head_width / 2)`). Output buffer seeded as a copy of input (not zero-allocated) before the rotation kernel writes.
- [ ] 3.7 `providers/cuda/src/kernels.rs`'s `CudaKernels::rope` gains the matching Rust parameter, updated launch config, and the same validation as 3.4.
- [ ] 3.8 `providers/cuda` test (real hardware): `head_count` absent/`1` bit-for-bit unchanged from today's existing `rope_matches_reference_cpu` conformance test. New conformance tests mirroring 3.5's three new Reference CPU cases (full multi-head, partial RoPE, GQA-shaped), each comparing CUDA's output against `providers/cpu`'s corrected implementation.
- [ ] 3.9 `CudaExecutor`'s "rope" dispatch arm reads a new `head_count` attribute, default `1` when absent. Mirror in `ReferenceCpuExecutor`'s "rope" arm.
- [ ] 3.10 `first_native_runtime.rs`: RoPE dispatch reads `head_count` via `node_attribute_u64(node, "head_count")`, falling back to the existing `qwen_rope_head_count(node, architecture)` id-suffix heuristic only when the attribute is absent (keeps any pre-existing hand-written test graph working).
- [ ] 3.11 Delete `dispatch_qwen_rope_per_head`; replace its call site with a single dispatch passing `head_count` and the full `NodeValue` input, no per-head slicing/reassembly in Rust.
- [ ] 3.12 Update the two large `#[cfg(test)]` oracle functions for the new single-dispatch RoPE shape.
- [ ] 3.13 Re-verify the per-node causal-evidence chain tests (`reach-architecture-freeze-1` task group 17) given RoPE's dispatch count per node changes from `head_count` to `1`.
- [ ] 3.14 Full regression: Reference CPU's own RoPE numeric output (prefill and decode) is unchanged end to end.

## 4. Weight edges thread `NodeValue`, shaped from `TensorEdge.descriptor` (design.md Decision 3)

- [ ] 4.1 `resolve_qwen_weight_edge`'s return type changes `HostTensor` -> `NodeValue`; resolve the weight edge's `TensorDescriptor` from the graph's own `TensorEdge.descriptor` (not the Provider, not Qwen config). `provider.read_tensor_value(resource_id)` maps `Host` -> `NodeValue::Host`, `Opaque` -> `NodeValue::Resident { id, shape: edge.descriptor.shape.dimensions.clone() }`.
- [ ] 4.2 Update the one call site (`dispatch_qwen_graph_node`'s weight-edge resolution) for the new return type.
- [ ] 4.3 Test: a weight resource reported `Opaque` by the resolved Provider resolves to `NodeValue::Resident` with the shape taken from the graph edge, without error (today's code hits `ResidencyUnavailable` here -- prove the fix with a regression test that fails without it).
- [ ] 4.4 Full regression: Reference CPU's own weight resolution (always `Host`) is unchanged.

## 5. `lm_head` tied embeddings: transpose once at Model Load (design.md Decision 5)

- [ ] 5.1 `WeightMaterializationTransaction::stage_weight` (or its caller, once per Model Load when `tied_embeddings` is set): after staging `token_embedding`, transpose it once via the existing `transpose_rows_cols` and stage the result under its own resource id through the same admission path every other weight uses.
- [ ] 5.2 `resolve_qwen_weight_edge`'s `lm_head` branch resolves the pre-transposed resource id directly (Decision 3's `NodeValue` path), removing the per-call Rust transpose from the hot path entirely.
- [ ] 5.3 Test: across multiple simulated generation steps reusing the same `ModelInstance`, `lm_head`'s weight is transposed exactly once (at Model Load), not once per step.
- [ ] 5.4 Full regression: `lm_head`'s numeric output (logits) is unchanged for both tied and non-tied-embeddings configurations.

## 6. Pre-admission rollback: a real transactional guarantee (design.md Decision 7)

- [ ] 6.1 Red test: demonstrate today's actual behavior when a Kernel dispatch fails after `MemoryManager::admit_kernel_output` already ran (before any fix), to have a concrete baseline for what "fixed" means.
- [ ] 6.2 Rollback on submit-time failure: pre-admit an output, force Kernel Registry/dispatch-plan construction to fail, assert the pre-admitted allocation is released and its `TensorResidency` removed.
- [ ] 6.3 Rollback on kernel/completion-time failure: pre-admit an output, force `submit_kernel`/completion itself to fail (reusing `TestFailableProvider`-style injection), assert the same -- no orphaned allocation, no falsely-`Ready`/`Active` residency, and the Provider's own `resource_allocations` tracking does not believe it self-admitted anything.
- [ ] 6.4 Verify Memory Manager and Provider storage are both checked in 6.2/6.3, not just one side.
- [ ] 6.5 Final test: after both failure scenarios, the Memory Manager's allocation count and residency table are exactly what they were before either attempt (zero net leak).

## 7. Real CUDA end-to-end proof (design.md Decision 8) -- required exit criterion

- [ ] 7.1 Build a minimal test fixture registering a real `CudaProvider`, resolving a `ModelInstance`/Prepared Plan onto it (real GPU, gated the same way this crate's other hardware-gated tests are).
- [ ] 7.2 Stage a small set of representative weights Device-resident through this fixture.
- [ ] 7.3 Dispatch one representative forward-pass-shaped chain: weight -> MatMul -> RMSNorm -> RoPE -> a projection, on real hardware.
- [ ] 7.4 Assert: no `ResidencyUnavailable` anywhere in the chain; no D2H/H2D between compatible chained Kernels beyond the chain's own final download; `ResourceAffinity`/`MemoryPlacement` correctly reflect CUDA/GPU0 throughout (via task group 1's production validation actually firing zero times); numeric result agrees with Reference CPU within tolerance; no allocation/residency leak after a successful run.
- [ ] 7.5 If this fixture's required scaffolding grows materially beyond reusing existing Prepared-Plan/`TestFailableProvider`-style test infrastructure, stop and report the actual size back rather than open-endedly expanding it (design.md's Risk note).

## 8. OpenSpec accuracy pass (audit P1-3)

- [ ] 8.1 Read `implement-cuda-provider-baseline/tasks.md` fully; correct any task text still describing pre-`enable-device-resident-kernel-chaining` (Host-resident storage, per-kernel-forced round-trip) behavior.
- [ ] 8.2 Confirm this change's own spec deltas (`operator`, `operator-scope`, `cuda-provider`, `resource-affinity`) match what actually shipped once tasks 1-7 land.

## 9. Cross-repository sequencing and full verification

- [ ] 9.1 `cargo build -p magnetar-runtime --lib`, `cargo test -p magnetar-runtime --lib`, `cargo clippy -p magnetar-runtime --lib --tests -- -D warnings`, `cargo fmt --check`, `cargo build --workspace`.
- [ ] 9.2 `cargo check --target wasm32-unknown-unknown -p magnetar-runtime --all-features`.
- [ ] 9.3 `providers/cpu` and `providers/cuda`: build, test, clippy, fmt clean (CUDA on real hardware, including `cargo deny --manifest-path providers/cuda/Cargo.toml check`).
- [ ] 9.4 Commit and push `providers/cpu`, then `providers/cuda`, then the parent, updating `SUBMODULES.md`'s compatibility matrix with the new commits.
- [ ] 9.5 Dispatch `gpu-runner-smoke.yml`, confirm green on the exact shipped commits (including task group 7's new E2E test).
- [ ] 9.6 `openspec validate make-first-native-cuda-hot-path-device-resident --strict`.
