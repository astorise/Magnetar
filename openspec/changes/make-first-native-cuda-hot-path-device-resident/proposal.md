## Why

`docs/audits/cuda-provider-audit-synthese-2026-09-06.md` approved the CUDA
Provider standalone but found three P0s blocking real CUDA use through
first-native, verified directly against the current code (not just the
audit's own description): weight edges force `TensorValue::into_host()`
unconditionally and fail with `ResidencyUnavailable` the moment a weight is
genuinely Device-resident; the shared Qwen dispatch helper
(`dispatch_reference_cpu_operator_multi`) hardcodes
`ResourceAffinity.provider = reference-cpu` even when the Prepared Plan
resolved CUDA/GPU0 (while the *placement* it computes alongside it correctly
says `Device`); and RMSNorm/RoPE force a Host round-trip between MatMul and
themselves. Two of these three are not new requirements -- `device-resident-
resource`'s "Same-Device Pipeline Avoids Mandatory Host Copy" (literally
scenario "MatMul to RMSNorm") and `resource-affinity`'s "Runtime-Native
Resource Affinity" ("the resource affinity records the selected Provider")
already mandate the correct behavior; first-native's implementation simply
does not conform to specs that already exist. Investigation during this
change's own design pass found RMSNorm's Host materialization is avoidable
today (both Providers' real `rmsnorm` kernels already accept a `[cols]`
weight and broadcast internally; first-native's manual Rust-side broadcast
is unneeded duplication), while RoPE's is a genuine Kernel-scope limit (the
existing `rope` Kernel only rotates the first `dimension` columns of a row,
so multi-head RoPE needs one Kernel call per head today) -- product decided
to close this by extending the `rope` Kernel's own scope to a native
`head_count` rather than leaving Rust-side per-head materialization as a
permanent exception.

## What Changes

- `resolve_qwen_weight_edge` becomes `NodeValue`-aware: a Device-resident
  weight passes through by resource id instead of forcing
  `TensorValue::into_host()`; still materializes to `HostTensor` for the
  one case that genuinely needs real bytes (tied-embedding `lm_head`
  transpose).
- `dispatch_reference_cpu_operator_multi` derives `ResourceAffinity`'s
  Provider/Device from the same Prepared Plan binding
  `resolved_output_placement`/`resolved_kernel_memory_class` already use,
  instead of hardcoding `REFERENCE_CPU_PROVIDER_NAME`. Adds a debug-time
  cross-check that `KernelInvocation`'s resolved provider/device and the
  `ResourceAffinity` attached to its resources agree.
- `dispatch_qwen_rmsnorm` drops its manual per-row weight broadcast and
  passes its input/weight through as `NodeValue` (Resident-eligible), the
  same pattern `dispatch_qwen_matmul`/`dispatch_qwen_attention` already use.
- The `rope` Kernel Operator (both `providers/cpu::rope` and
  `providers/cuda::CudaKernels::rope`, plus each `CudaExecutor`/
  `ReferenceCpuExecutor` "rope" dispatch arm and `kernels.cu`'s
  `rope_kernel`) gains a `head_count` parameter (default/existing behavior
  when `head_count == 1`, backward compatible): the per-head rotation loop
  moves inside the Kernel (GPU-parallel for CUDA, a single Rust loop for
  Reference CPU) instead of `first_native_runtime.rs` slicing per head,
  dispatching a Kernel invocation per head, and reassembling in Rust.
  `dispatch_qwen_rope_per_head` is replaced by a single dispatch per RoPE
  node, `NodeValue`-aware like the other operators.
- `implement-cuda-provider-baseline/tasks.md` updated to match the
  post-`enable-device-resident-kernel-chaining`/`unify-provider-output-
  admission-and-residency` reality (some task text still describes the old
  Host-resident-storage/per-kernel-round-trip state).
- Regression test: pre-admit an output, force the Kernel dispatch to fail,
  assert the Memory Manager holds no orphaned allocation and no
  Ready/Active residency record for it (transactional rollback proof for
  `unify-provider-output-admission-and-residency`'s pre-admission
  mechanism, requested by the new audit).

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `cuda-provider`: `CudaKernels::rope`'s signature gains `head_count`; new
  scenario describing native multi-head rotation without a Host round-trip
  or per-head Kernel dispatch.
- `operator-scope`: "RoPE Scope" gains a scenario permitting a single
  Kernel invocation to rotate multiple heads via `head_count`, rather than
  one invocation per head being the only supported shape.

`device-resident-resource`'s "Same-Device Pipeline Avoids Mandatory Host
Copy" and `resource-affinity`'s "Runtime-Native Resource Affinity" are
**not** modified -- this change brings the implementation into conformance
with what they already require; no wording changes needed there.

## Impact

- `magnetar-runtime/src/first_native_runtime.rs` (weight edge resolution,
  shared dispatch helper's affinity construction, RMSNorm dispatch, RoPE
  dispatch and its per-head helper's removal).
- `magnetar-runtime/src/tests.rs` / `first_native_runtime/tests.rs`
  (updated oracles, new rollback regression test).
- `providers/cpu/src/lib.rs` (`rope`'s new `head_count` parameter).
- `providers/cuda/src/kernels.rs`, `providers/cuda/src/kernels.cu`,
  `providers/cuda/src/executor.rs` (`rope`'s new `head_count` parameter,
  CUDA kernel's per-head thread indexing, "rope" dispatch arm's new
  attribute).
- `openspec/specs/cuda-provider/specs/...` delta (this change),
  `openspec/specs/operator-scope` delta (this change),
  `openspec/changes/implement-cuda-provider-baseline/tasks.md` (accuracy
  fix, not new tasks).
