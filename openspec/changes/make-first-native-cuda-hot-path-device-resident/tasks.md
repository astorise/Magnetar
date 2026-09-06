## 1. ResourceAffinity conformance fix (design.md Decision 2)

- [ ] 1.1 Add `resolved_resource_affinity(prepared_plan, node, execution_context) -> ResourceAffinity` to `first_native_runtime.rs`, mirroring `resolved_output_placement`'s lookup exactly.
- [ ] 1.2 `dispatch_reference_cpu_operator_multi` calls it in place of its hardcoded `ResourceAffinity::new(...).with_provider(ProviderBinding::new(REFERENCE_CPU_PROVIDER_NAME))`.
- [ ] 1.3 Add a `debug_assert!` cross-check after `KernelDispatchPlan` is built: the resolved invocation's Provider/Device agrees with the `affinity` used for its resources.
- [ ] 1.4 Test: a Qwen node whose Prepared Plan binds a non-Reference-CPU Provider/Device produces `ResourceAffinity` matching that binding, not `reference-cpu`.
- [ ] 1.5 Full regression: confirm every existing Reference-CPU-only test (no Prepared Plan, or a Plan binding Reference CPU) is bit-for-bit unaffected.

## 2. RMSNorm Host materialization removed (design.md Decision 3)

- [ ] 2.1 Confirm (already verified during design, re-confirm against current code before editing) that `providers/cpu::rmsnorm`/`providers/cuda::CudaKernels::rmsnorm` both accept a `[cols]`-shaped weight and broadcast internally.
- [ ] 2.2 `dispatch_qwen_rmsnorm`'s signature changes `input: HostTensor, weight: HostTensor` to `input: NodeValue, weight: NodeValue`; delete the manual per-row weight broadcast.
- [ ] 2.3 `dispatch_qwen_graph_node`'s "rmsnorm" arm stops calling `.into_host()` on either input before dispatch; both become `NodeInputResource::Resident`-eligible in `dispatch_reference_cpu_operator_multi`.
- [ ] 2.4 Update the two large `#[cfg(test)]` oracle functions (`execute_qwen_prefill/decode_hidden_states_through_dispatch`) for the new signature.
- [ ] 2.5 Test: a MatMul output already Resident under the resolved Provider passes into RMSNorm without an intervening `.into_host()` call (extend the existing `OpaqueReportingExecutor`-based unit test style from `unify-provider-output-admission-and-residency`).
- [ ] 2.6 Full regression: Reference CPU's own RMSNorm numeric output is unchanged (it never returns `Opaque`, so this is a pure plumbing change for it).

## 3. Weight edges thread `NodeValue` (design.md Decision 1)

- [ ] 3.1 `resolve_qwen_weight_edge`'s return type changes `HostTensor` -> `NodeValue`; `provider.read_tensor_value(resource_id)` maps to `NodeValue::Host`/`NodeValue::Resident` instead of immediately calling `.into_host()`.
- [ ] 3.2 The `lm_head`/tied-embeddings branch materializes via `.into_host()` itself (still needs real bytes for `transpose_rows_cols`), scoped to exactly that branch.
- [ ] 3.3 Update the one call site (`dispatch_qwen_graph_node`'s weight-edge resolution) for the new return type.
- [ ] 3.4 Test: a weight resource reported `Opaque` by the resolved Provider resolves to `NodeValue::Resident` without error (today's code would hit `ResidencyUnavailable` here -- prove the fix with a regression test that fails without it, same rigor as the Memory Manager leak fix earlier this session).
- [ ] 3.5 Full regression: Reference CPU's own weight resolution (always `Host`) is unchanged.

## 4. RoPE Kernel gains native `head_count` (design.md Decision 4)

- [ ] 4.1 `providers/cpu::rope` gains `head_count: u64`; loop one extra level, rotate `[head * dimension, head * dimension + dimension)` per row per head. Validate `head_count >= 1` and `head_count * dimension == cols`.
- [ ] 4.2 `providers/cpu` test: `head_count = 1` reproduces today's exact output (regression, not just new coverage); new test for `head_count > 1` against a hand-computed expected rotation.
- [ ] 4.3 `providers/cuda/src/kernels.cu`'s `rope_kernel` gains `head_count`; re-derive `row`/`head`/`pair`/`col_base` from the flattened thread index as design.md's Decision 4 specifies. `providers/cuda/src/kernels.rs`'s `CudaKernels::rope` gains the matching Rust parameter, updated launch config (`rows * head_count * half` threads), and the same validation as 4.1.
- [ ] 4.4 `providers/cuda` test (real hardware): `head_count = 1` bit-for-bit unchanged from today's existing `rope_matches_reference_cpu` conformance test; new multi-head conformance test comparing CUDA's multi-head output against `providers/cpu::rope`'s multi-head output for the same input.
- [ ] 4.5 `CudaExecutor`'s "rope" dispatch arm reads a new `head_count` attribute, default `1` (`Self::attribute_u64(&invocation.attributes, "head_count", 1)` or equivalent) so existing callers without it are unaffected. Mirror in `ReferenceCpuExecutor`'s "rope" arm.
- [ ] 4.6 `first_native_runtime.rs`: delete `dispatch_qwen_rope_per_head`; replace its call site with a single dispatch passing `head_count` (from `architecture.attention_head_count`) and the full `NodeValue` input, no per-head slicing/reassembly in Rust.
- [ ] 4.7 Update the two large `#[cfg(test)]` oracle functions for the new single-dispatch RoPE shape.
- [ ] 4.8 Re-verify the per-node causal-evidence chain tests (`reach-architecture-freeze-1` task group 17) given RoPE's dispatch count per node changes from `head_count` to `1`.
- [ ] 4.9 Full regression: Reference CPU's own RoPE numeric output (prefill and decode) is unchanged end to end.

## 5. Pre-admission rollback regression test (audit P1-4 / design.md Decision 5)

- [ ] 5.1 Test: pre-admit an output via `MemoryManager::admit_kernel_output`, force the Kernel dispatch to fail (reuse existing failure-injection pattern, e.g. `TestFailableProvider`-style), assert no orphaned allocation and no falsely-`Ready`/`Active` residency for the pre-admitted id afterward.

## 6. OpenSpec accuracy pass (audit P1-3 / design.md Decision 6)

- [ ] 6.1 Read `implement-cuda-provider-baseline/tasks.md` fully; correct any task text still describing pre-`enable-device-resident-kernel-chaining` (Host-resident storage, per-kernel-forced round-trip) behavior.
- [ ] 6.2 Confirm this change's own spec deltas (`cuda-provider`, `operator-scope`) match what actually shipped once tasks 1-4 land.

## 7. Cross-repository sequencing and full verification

- [ ] 7.1 `cargo build -p magnetar-runtime --lib`, `cargo test -p magnetar-runtime --lib`, `cargo clippy -p magnetar-runtime --lib --tests -- -D warnings`, `cargo fmt --check`, `cargo build --workspace`.
- [ ] 7.2 `cargo check --target wasm32-unknown-unknown -p magnetar-runtime --all-features`.
- [ ] 7.3 `providers/cpu` and `providers/cuda`: build, test, clippy, fmt clean (CUDA on real hardware, including `cargo deny --manifest-path providers/cuda/Cargo.toml check`).
- [ ] 7.4 Commit and push `providers/cpu`, then `providers/cuda`, then the parent, updating `SUBMODULES.md`'s compatibility matrix with the new commits.
- [ ] 7.5 Dispatch `gpu-runner-smoke.yml`, confirm green on the exact shipped commits.
- [ ] 7.6 `openspec validate make-first-native-cuda-hot-path-device-resident --strict`.
