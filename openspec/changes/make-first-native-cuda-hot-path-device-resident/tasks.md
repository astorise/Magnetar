## 1. ResourceAffinity: resolved helper, Resident preservation, production validation (design.md Decision 6)

- [x] 1.1 Add `resolved_resource_affinity(prepared_plan, node, execution_context) -> ResourceAffinity` to `first_native_runtime.rs`, mirroring `resolved_output_placement`'s lookup exactly.
- [x] 1.2 `dispatch_reference_cpu_operator_multi` calls it for `NodeInputResource::Fresh` resources and its own new outputs (replacing the hardcoded `ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME)`).
- [x] 1.3 For `NodeInputResource::Resident` resources, look up the resource's existing recorded `ResourceAffinity` (via its `TensorResidency`/Memory Manager record) and aggregate it with the dispatch's own affinity through the existing `resource-affinity` aggregation contract, rejecting on conflict rather than overwriting. Extracted as `resident_resource_affinity` for direct testability.
- [x] 1.4 Added the production-level (not `debug_assert!`) structured validation at the Kernel/Runtime dispatch boundary, extracted as `validate_invocation_provider_matches_affinity` (also for direct testability): after `KernelDispatchPlan` is built, rejects with `InferenceApiError::GraphPlanningFailed` if `invocation.kernel.provider` disagrees with the affinity's own provider. Real, not hypothetical: `PlanNodeBinding::new` accepts its `kernel: KernelId` and its own `provider: ProviderBinding` as independent parameters with no validation they agree (confirmed by reading it).
- [x] 1.5 Test: `resolved_resource_affinity_matches_a_non_reference_cpu_plan_binding` / `..._falls_back_to_reference_cpu_without_a_binding`.
- [x] 1.6 Test: `resident_resource_affinity_is_preserved_not_overwritten` (a Device binding absent from the dispatch's own affinity survives aggregation) / `resident_resource_affinity_conflict_is_rejected`.
- [x] 1.7 Test: `validate_invocation_provider_matches_affinity_rejects_a_real_divergence` / `..._accepts_agreement`.
- [x] 1.8 Full regression: 1201/1201 `magnetar-runtime` tests pass (1195 + 6 new), clippy/fmt clean.

## 2. RMSNorm Host materialization removed (design.md Decision 4)

- [x] 2.1 Confirmed: `providers/cpu::rmsnorm`/`providers/cuda::CudaKernels::rmsnorm` both derive `cols` from the input's own shape and accept a `[cols]`-shaped weight, broadcasting internally.
- [x] 2.2 `dispatch_qwen_rmsnorm`'s signature changed `input: HostTensor, weight: HostTensor` to `input: NodeValue, weight: NodeValue`; manual per-row weight broadcast deleted.
- [x] 2.3 `dispatch_qwen_graph_node`'s "rmsnorm" arm stops calling `.into_host()` on either input before dispatch.
- [x] 2.4 Updated all 6 call sites across the two large `#[cfg(test)]` oracle functions for the new signature; one of them (`post_attention_host`) was itself an unneeded `.into_host()` this fix removed outright.
- [x] 2.5 Test: `rmsnorm_accepts_a_resident_input_without_materializing_it_first` -- a MatMul-shaped output already Resident under `OpaqueReportingExecutor` passes into RMSNorm and computes the correct real result (RMS-normalization of `[2,4,4,8]` -> `[0.4,0.8,0.8,1.6]`), previously impossible to even call with a Resident value.
- [x] 2.6 Full regression: 1202/1202 `magnetar-runtime` tests pass (1201 + 1 new), including every existing Reference CPU forward-pass test unchanged; clippy/fmt clean.

## 3. RoPE: explicit `head_count` graph attribute, corrected multi-head semantics (design.md Decisions 1-2)

- [x] 3.1 `operator.rs`'s `"rope"` `OperatorAttributeSchema` gains `head_count` as `OperatorAttributeRule::optional(OperatorAttributeKind::Integer)`.
- [x] 3.2 Schema test: `rope_schema_accepts_head_count_with_correct_kind_only` -- absent/`1`/`>1` accepted, wrong kind rejected, a still-unknown attribute name still rejected.
- [x] 3.3 `qwen_model_component.rs`'s `rope_q` node construction adds `head_count = a.attention_head_count`; `rope_k` adds `head_count = a.kv_head_count`.
- [x] 3.4 `providers/cpu::rope` gains `head_count: u64` with the corrected `head_width = cols / head_count` semantics and copy-seeded output.
- [x] 3.5 `providers/cpu` tests: regression (`head_count` absent/`1`) plus new multi-head, partial-RoPE, and GQA-shaped cases, each cross-checked against independent single-head slices.
- [x] 3.6 `providers/cuda/src/kernels.cu`'s `rope_kernel` gains `head_count`, corrected `(row, head, pair)` indexing, device-to-device copy seed (`clone_dtod`, replacing `alloc_zeros`).
- [x] 3.7 `providers/cuda/src/kernels.rs`'s `CudaKernels::rope` gains the matching parameter, updated launch config (`rows * head_count * half`), and the same validation as 3.4.
- [x] 3.8 `providers/cuda` tests (real hardware, RTX 3070 Ti): `rope_matches_reference_cpu` (unchanged), plus new `rope_multi_head_matches_reference_cpu`/`rope_partial_rotation_matches_reference_cpu`/`rope_gqa_shaped_head_count_matches_reference_cpu` -- all pass against `providers/cpu`'s corrected implementation.
- [x] 3.9 `CudaExecutor`'s and `ReferenceCpuExecutor`'s (`providers/cpu`) "rope" dispatch arms read `head_count`, default `1`.
- [x] 3.10 `first_native_runtime.rs`: RoPE dispatch reads `head_count` via `node.attributes.get("head_count")`, falling back to `qwen_rope_head_count(node, architecture)` only when absent.
- [x] 3.11 Deleted `dispatch_qwen_rope_per_head`; replaced with single-dispatch `dispatch_qwen_rope`. `output_target` deliberately stays `None` for this arm (matches the pre-existing, unchanged top-of-function comment: RoPE's KV-cache-tracked output edge is handled by `execute_qwen_graph_nodes`'s own unconditional path for this node kind, not pre-admission).
- [x] 3.12 Updated all 6 oracle call sites (4 rope calls + 2 downstream materializations) across the two large `#[cfg(test)]` oracle functions.
- [x] 3.13 Re-verified: the full 1208-test suite includes the per-node causal-evidence chain tests unchanged.
- [x] 3.14 Full regression, with a genuine bug found and fixed along the way (see below): `magnetar-runtime` 1208/1208, `providers/cpu` (own suite), `providers/cuda` 27/27 real hardware, clippy/fmt/wasm32 clean.

**Critical bug found and fixed during 3.14's regression pass, not by inspection but by a real end-to-end oracle divergence**: `magnetar-runtime/src/reference_cpu.rs` carries its own in-crate copy of `rope()` (the "in-crate test double" the module's own doc comment describes, independent from `providers/cpu`'s copy) -- and its "rope" Kernel dispatch arm is what `ReferenceCpuExecutor` (the executor `dispatch_qwen_graph_node` and every oracle actually dispatch through) runs. Tasks 3.4-3.9 above updated `providers/cpu` and `providers/cuda`'s copies and dispatch arms, but missed this third copy entirely -- it kept its old 5-argument signature and silently ignored the new `head_count` attribute, rotating only the first `dimension` columns of every row regardless of `head_count`, leaving every head beyond the first completely unrotated. This produced identical (but wrong) results whether compared through `dispatch_qwen_graph_node` (the graph) or through the oracle functions this task group also updated to call the same shared `dispatch_qwen_rope` -- both shared the same silent bug, so they agreed with each other while disagreeing with `e2e_forward_hidden_states`'s fully independent, hand-written reference implementation (`apply_rope_per_head`, deliberately left untouched), which is what caught it. Fixed by applying the identical `head_count`/`head_width` correction to `reference_cpu.rs`'s own `rope()` and its dispatch arm, plus the same new multi-head/partial/GQA test trio in `reference_cpu/tests.rs`. This is the exact class of gap task group 7's end-to-end hardware test exists to catch -- confirmed here at the Reference CPU level before ever reaching real CUDA hardware.

## 4. Weight edges thread `NodeValue`, shaped from `TensorEdge.descriptor` (design.md Decision 3)

- [x] 4.1 `resolve_qwen_weight_edge`'s return type changed `HostTensor` -> `NodeValue`; resolves the weight edge's `TensorDescriptor` from the graph's own `TensorEdge.descriptor` (new parameter), not the Provider, not Qwen config. `provider.read_tensor_value(resource_id)` maps `Host` -> `NodeValue::Host`, `Opaque` -> `NodeValue::Resident { id, shape: descriptor.shape.dimensions.clone() }`. `lm_head`'s tied-embeddings branch still materializes for its Rust-side transpose (task group 5 removes this).
- [x] 4.2 Updated the one call site (`execute_qwen_graph_nodes`'s weight-edge resolution, which now also looks up the edge for its descriptor). Also fixed a related gap found in the same area: `passthrough_eligible`'s list still excluded `rmsnorm`/`rope` even though task groups 2-3 already made both `NodeValue`-based -- added them, so a Resident edge feeding either no longer gets forced to materialize one level up before ever reaching their own (already-fixed) dispatch functions.
- [x] 4.3 Test: `weight_edge_resolves_opaque_weight_to_resident_without_materializing` -- an `Opaque` weight resolves to `NodeValue::Resident` with the graph edge's shape, not an error.
- [x] 4.4 Full regression: 1209/1209 `magnetar-runtime` tests pass (1208 + 1 new), clippy/fmt clean.

## 5. `lm_head` tied embeddings: transpose once at Model Load (design.md Decision 5)

- [x] 5.1 New shared helper `qwen_weights_with_derived_lm_head` (not inside `WeightMaterializationTransaction` itself, which stays generic per Correctif 9): for a tied-embeddings fixture with no explicit `lm_head` entry, transposes `token_embedding` once and inserts the result under the name `lm_head`, before the weight map ever reaches `materialize_model_instance_weights`. Called from `bind_qwen_fixture_weights` and `load_fixture_instance_with_weights` (both weight-loading entry points this fixture has).
- [x] 5.2 `resolve_qwen_weight_edge` no longer has *any* `lm_head`/`tied_embeddings` special case -- its `tied_embeddings: bool` parameter was removed entirely; `lm_head` now resolves exactly like any other weight, by name, already pre-transposed. `weight_bindings` genuinely contains an `lm_head` entry from Model Load onward.
- [x] 5.3 Test: `lm_head_weight_is_transposed_once_at_model_load_not_per_generation_step` -- runs a real prefill then decode dispatch against the same `ModelInstance`, confirms the `lm_head` resource id and its staged data (byte-identical to `token_embedding`'s independently-computed transpose) are unchanged after both dispatches.
- [x] 5.4 Full regression: 1210/1210 `magnetar-runtime` tests pass (1209 + 1 new), including the existing weight-sensitivity/tied- and non-tied-embeddings tests, unchanged; clippy/fmt clean.

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
