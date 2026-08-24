# Design Notes

## Tensor storage is Provider-owned and opaque

The Runtime's existing contracts (`compute.rs`, `planning.rs`, `kernel.rs`,
`memory.rs`) model tensor resources purely as identity and accounting:
`TensorResourceDescriptor`, `ExecutionInput`/`ExecutionOutput`, and
`KernelResource` carry a `TensorResourceId` and a `TensorDescriptor`, never
raw bytes. No Provider in the codebase (including the `TestProvider` fixture)
holds or exposes actual host/device buffers through the Runtime; buffer
ownership is Provider-internal, matching the existing convention that
"Provider-owned resources SHALL remain opaque" (see the Adapter contract).

The Reference CPU Provider follows this convention rather than inventing a
new cross-cutting byte-buffer abstraction:

- `ReferenceCpuExecutor` holds the actual numeric data (`HostTensor`, a
  `Vec<f32>` with shape) behind a `TensorResourceId`-keyed store. The
  Runtime, Kernel Registry, and Kernel Dispatch layers never see this
  storage — only kernel advertisements, invocations, and results.
- Kernel math (`matmul`, `embedding_lookup`, `rmsnorm`, `rope`, `attention`,
  `softmax_rows`, `silu`, `gelu`, `add`, `mul`, `residual_add`,
  `dtype_conversion`, `layout_conversion`) is implemented as pure,
  independently unit-tested functions over `HostTensor`, so correctness can
  be verified with known-input/known-output tests without requiring a full
  Execution Graph interpreter.
- `ReferenceCpuExecutor::execute_invocation` is the Kernel Contract-level
  entry point: it validates a Runtime-created `KernelInvocation` against the
  advertised `KernelAdvertisement` and the portable `OperatorSpec`, then
  dispatches to the matching pure kernel function and records outputs back
  into opaque storage.
- `ReferenceCpuExecutor` also implements `ProviderExecutionApi` for the
  legacy `ComputeGraph`/`ComputeExecutionPlan` path, at the same fidelity
  every other Provider in this codebase currently implements it (bookkeeping
  only — no other Provider executes real numeric work through that API
  either, since `ComputeExecutionPlan` only carries a `ComputeGraphId`
  reference plus resource descriptors, not tensor bytes or the graph body).

## Attention: grouped-query and sliding-window support

`attention` supports `kv_head_count` (grouped-query attention: `head_count`
must be an exact multiple of `kv_head_count`, with each group of query heads
sharing one key/value head) and `window_size` (sliding-window attention:
each query only attends to its most recent `window_size` keys, composing
with causal masking). Both read from the portable `attention` Operator's
existing attribute schema (`kv_head_count`, `window_size` are already
defined there) — no Operator catalog changes were needed. An arbitrary mask
tensor input was not added, since the shared Operator schema fixes attention
at exactly 3 inputs (q/k/v); doing so would be a cross-cutting Operator
catalog change beyond this Provider's scope.

## Conformance scope: Kernel Contract only, not the legacy Compute capability

The codebase carries two largely-separate Provider execution surfaces: the
newer Kernel/Operator Contract this Provider implements, and an older
`ComputeGraph`/`ProviderComputeAdvertisement` capability system (`compute.rs`,
`scheduler.rs`) with its own `ProviderCompute`/`ProviderDataMovement`
conformance profiles. Reference CPU deliberately does not populate
`ProviderMetadata.compute_advertisement`, so those profiles are correctly
skipped by `ProviderConformanceSuite` rather than exercised against a system
this Provider doesn't implement. Dtype/layout/memory correctness is instead
validated through this Provider's own kernel-level tests
(`dtype_conversion`, `layout_conversion`, and the Memory Manager integration
tests).

## Extending shared types where the task list required it

A later pass completed the remaining tasks, including several that need
small, additive changes to shared types used by every Provider. Each was
verified to be safe before making it:

- `DeviceMetadata` (`device.rs`) gained `dtype_support`, `layout_support`,
  `memory_class_support`, `execution_limits: DeviceExecutionLimits`, and
  `pressure: ProviderPressureLevel`. All are additive fields with defaults
  set in `DeviceMetadata::new`; the one other construction site
  (`kernel_dispatch.rs`, a struct literal) was updated to set them to their
  defaults too. No existing behavior changes for any other Provider.
- `KernelObservationKind` (`kernel.rs`) gained `ProviderRegistered` and
  `DeviceDetected` variants. Checked first that nothing in the codebase
  matches this enum exhaustively (only `Debug` derives consume it), so the
  addition can't break exhaustiveness elsewhere.
- The portable `rope` Operator's attribute schema already defined an
  optional `position_mode` string (found on closer reading of
  `operator.rs`, no catalog change was actually needed); `attention`'s
  schema already defined `kv_head_count`, `window_size`, and
  `attention_mask_kind` similarly. Reference CPU now validates all of them.
- A generic `activation` Operator/Kernel (kind `"silu"`/`"gelu"`, dispatched
  on the schema's existing required `kind` attribute) was added alongside
  the dedicated `silu`/`gelu` kernels, giving "activation kind" a real
  attribute to validate against.
- `attention` now advertises a required Host workspace and requests it
  through `ReferenceCpuExecutor::allocate_workspace` (Memory-Manager-backed);
  the shared `validate_invocation` already rejects a missing workspace, so
  "validate workspace references" needed no new Provider-side check.

## Deliberately not extended

- An arbitrary attention *mask tensor* input was not added: the shared
  `attention` Operator schema fixes arity at exactly 3 inputs (q/k/v), and
  changing that would affect every Provider implementing `attention`, not
  just this one. `attention_mask_kind` (a string selector already in the
  schema) covers the two mask shapes expressible without a new tensor
  input instead.
- Incremental/paged KV cache is not implemented. Rather than leaving
  `KernelAdvertisement.kv_cache` absent, it is `Some(KernelKvCacheMetadata)`
  with every support flag explicitly `false` — a stated fact, not a gap.
- Quantized formats (`quantize`/`dequantize`) are not advertised or
  implemented; invoking them returns an explicit
  `reference-cpu-dtype-unsupported` error rather than a silent fallback.
- Paged/blocked/strided/opaque layouts and non-`f32` dtypes are rejected
  explicitly rather than silently converted, since Reference CPU's internal
  storage only models contiguous `f32` tensors.
