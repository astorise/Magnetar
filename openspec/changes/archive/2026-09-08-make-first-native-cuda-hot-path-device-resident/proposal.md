## Why

`docs/audits/cuda-provider-audit-synthese-2026-09-06.md` approved the CUDA
Provider standalone but found three P0s blocking real CUDA use through
first-native, verified directly against the current code: weight edges force
`TensorValue::into_host()` unconditionally and fail with
`ResidencyUnavailable` the moment a weight is genuinely Device-resident; the
shared Qwen dispatch helper hardcodes `ResourceAffinity.provider =
reference-cpu` even when the Prepared Plan resolved CUDA/GPU0; and
RMSNorm/RoPE force a Host round-trip between MatMul and themselves. Two of
these three are not new requirements -- `device-resident-resource`'s
"Same-Device Pipeline Avoids Mandatory Host Copy" and `resource-affinity`'s
"Runtime-Native Resource Affinity" already mandate the correct behavior;
first-native's implementation simply does not conform to specs that already
exist.

A pre-implementation product review of this change's first design pass
(`docs/audits/revue-openspec-cuda-pre-implementation-2026-09-06.md`) returned
**NO-GO**: the RoPE fix as first designed was wrong for GQA/MQA models (Q and
K need different head counts, and the design's validation rule broke partial
RoPE), the new `head_count` attribute was missing from the actual
`OperatorAttributeSchema` that would reject it at validation time, the
Device-resident weight fix had no defined source for a `NodeValue::Resident`
value's shape (`TensorValue::Opaque` carries none), the `lm_head` tied-
embedding case kept a hot-path Host round-trip the review found too costly,
`ResourceAffinity`'s consistency check was dev-only (`debug_assert!`) with no
production enforcement, Resident resources (like weights) would have their
recorded affinity silently overwritten instead of preserved, the pre-
admission rollback was scoped as a single test rather than a real
transactional guarantee, and a genuine CUDA end-to-end proof was deferred
rather than required before this change could be considered complete. This
revision incorporates all ten of the review's requested decisions before any
implementation starts.

## What Changes

- `resolve_qwen_weight_edge` becomes `NodeValue`-aware: a Device-resident
  weight passes through by resource id instead of forcing
  `TensorValue::into_host()`, with its shape sourced from the graph's own
  `TensorEdge.descriptor` (never downloaded, never recomputed from Qwen
  config) -- still materializes to `HostTensor` for the one case that
  genuinely needs real bytes today (see the `lm_head` decision below, which
  removes that case from the hot path).
- The Qwen graph builder (`qwen_model_component.rs`, the actual first-native
  graph source) sets an explicit `head_count` attribute on each RoPE node at
  graph-construction time: `attention_head_count` for Q, `kv_head_count` for
  K. First-native's RoPE dispatch reads this attribute instead of inferring
  it from the consuming node's id suffix.
- The `rope` Kernel Operator (`providers/cpu::rope`,
  `providers/cuda::CudaKernels::rope`, `kernels.cu`'s `rope_kernel`, each
  executor's "rope" dispatch arm, and `operator.rs`'s `OperatorAttributeSchema`
  for `"rope"`) gains an optional `head_count` attribute with corrected,
  GQA- and partial-RoPE-safe semantics: `head_width = cols / head_count`,
  each head's own rotation width `dimension <= head_width` (already-required
  invariant, now actually validated), rotating only
  `[head * head_width, head * head_width + dimension)` per head. `head_count`
  absent or `1` reproduces today's exact single-block behavior bit-for-bit.
  The per-head Rust dispatch loop (`dispatch_qwen_rope_per_head`) is
  replaced by a single dispatch per RoPE node.
- `dispatch_reference_cpu_operator_multi` derives `ResourceAffinity` from the
  same Prepared Plan binding `resolved_output_placement`/
  `resolved_kernel_memory_class` already use, instead of hardcoding
  `REFERENCE_CPU_PROVIDER_NAME` -- but only for **freshly admitted**
  resources. A resource that is already Resident (a weight, a prior node's
  output) keeps its own recorded affinity, aggregated with the dispatch's
  affinity through the existing `resource-affinity` aggregation contract
  (conflict rejected, not silently overwritten). The Provider/Device
  consistency check that catches a future regression here is enforced in
  production (a structured, always-on validation at the Kernel/Runtime
  dispatch boundary), not only in debug builds.
- `dispatch_qwen_rmsnorm` drops its unneeded manual per-row weight broadcast
  (both real Kernels already accept `[cols]` and broadcast internally) and
  passes its input/weight through as `NodeValue`.
- `lm_head`'s tied-embedding weight is transposed **once**, at Model Load
  time, into its own cached Device-resident resource, instead of physically
  transposing it in Rust on every generation step's `lm_head` dispatch --
  removing the last permanent Host round-trip from the hot path without
  requiring a new `transpose_b`-aware shape rule on the portable `matmul`
  Operator (deliberately not pursued this change: a second, independent
  Operator-contract change, avoidable here).
- Pre-admission rollback (`unify-provider-output-admission-and-residency`'s
  mechanism) gets a real transactional guarantee, not just a single test:
  explicit rollback behavior for a submit-time failure and for a
  kernel/completion-time failure, each verified to leave no orphaned Memory
  Manager allocation and no falsely-`Ready`/`Active` residency record.
- A real, hardware-verified first-native CUDA end-to-end run (weights
  Device-resident, MatMul -> RMSNorm -> RoPE -> projection all on GPU0, one
  final download) becomes a required exit criterion for this change, not a
  deferred follow-up.
- `implement-cuda-provider-baseline/tasks.md` updated to match the
  post-`enable-device-resident-kernel-chaining`/`unify-provider-output-
  admission-and-residency` reality.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `operator`: "RoPE Operator" requirement extended to also represent
  `head_count` when a model's RoPE application spans multiple heads with
  independent rotation.
- `operator-scope`: "RoPE Scope" gains multi-head, GQA-aware, partial-RoPE
  scenarios (corrected from this change's first draft).
- `cuda-provider`: `CudaKernels::rope`'s signature gains `head_count` with
  the corrected `head_width = cols / head_count` semantics.
- `resource-affinity`: "Affinity Constraint Aggregation" gains a scenario
  for preserving a Resident resource's own affinity rather than overwriting
  it with a dispatch's freshly-derived one; "Runtime-Native Resource
  Affinity" gains a scenario requiring production (not only debug-time)
  enforcement that a `KernelInvocation`'s resolved Provider/Device agrees
  with the `ResourceAffinity` recorded for its resources.

`device-resident-resource`'s "Same-Device Pipeline Avoids Mandatory Host
Copy" is **not** modified -- this change brings the implementation into
conformance with what it already requires.

## Impact

- `magnetar-runtime/src/first_native_runtime.rs` (weight edge resolution,
  shared dispatch helper's affinity construction and Resident-preservation
  logic, RMSNorm dispatch, RoPE dispatch and its per-head helper's removal,
  `lm_head` weight resolution, a new production affinity-consistency
  validation).
- `magnetar-runtime/src/qwen_model_component.rs` (RoPE node construction
  gains `head_count`; a new one-time transposed-weight staging step for
  tied embeddings).
- `magnetar-runtime/src/operator.rs` (`"rope"` operator's
  `OperatorAttributeSchema` gains `head_count`).
- `magnetar-runtime/src/tests.rs` / `first_native_runtime/tests.rs`
  (updated oracles, new regression tests: GQA head_count, partial RoPE,
  weight Resident passthrough, rollback on submit/kernel failure).
- `providers/cpu/src/lib.rs` (`rope`'s new `head_count` parameter,
  corrected indexing).
- `providers/cuda/src/kernels.rs`, `providers/cuda/src/kernels.cu`,
  `providers/cuda/src/executor.rs` (`rope`'s new `head_count` parameter,
  corrected CUDA kernel indexing, "rope" dispatch arm's new attribute).
- A new first-native + real `CudaProvider` end-to-end test fixture (real
  GPU, `arc-gpu-magnetar`), scoped in tasks.md.
- `openspec/specs/operator`, `openspec/specs/operator-scope`,
  `openspec/specs/cuda-provider`, `openspec/specs/resource-affinity` deltas
  (this change), `openspec/changes/implement-cuda-provider-baseline/tasks.md`
  (accuracy fix, not new tasks).
