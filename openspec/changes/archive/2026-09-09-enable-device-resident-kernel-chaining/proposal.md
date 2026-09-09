## Why

The full CUDA Provider audit (2026-09-05) confirms three P0 findings by direct
code inspection, and they are more severe than this repository's own
`implement-cuda-provider-baseline` change already admits as non-goals:

1. `providers/cuda/src/kernels.rs` performs implicit host-to-device and
   device-to-host copies inside every kernel method (`clone_htod` ->
   compute -> `clone_dtoh`), directly contradicting the already-shipped
   "CUDA Provider Explicit Data Movement" requirement in
   `implement-cuda-provider-baseline`'s own spec delta.
2. `providers/cuda/src/executor.rs` stores every Tensor Resource as a
   host-resident `HostTensor` in a `Mutex<BTreeMap>`, while
   `write_tensor_admitted`/`execute_invocation_with_memory_manager` report
   `MemoryPlacement::Device` residency to the Runtime Memory Manager --
   contradicting that same change's "CUDA Provider Memory Manager
   Integration" requirement ("SHALL allocate device memory per Tensor
   Resource directly"). The ledger claims device residency that does not
   physically exist.
3. `magnetar-runtime/src/first_native_runtime.rs`'s per-node dispatch loop
   (`execute_qwen_graph_nodes`, the sole production Qwen execution path)
   calls `TensorValue::into_host` on every node's every input edge, at every
   generation step -- not only at the three documented boundaries (weight
   binding, KV-history concatenation, final logits extraction). Two
   compatible, consecutive Device-resident Kernels can never chain without a
   host round-trip today, regardless of which Provider is bound, directly
   contradicting the already-established `device-resident-resource`
   capability's "Same-Device Pipeline Avoids Mandatory Host Copy"
   requirement. The root cause is not a missing completion/async primitive
   (`ProviderExecutionHandle`/`status`/`complete` already exist and already
   satisfy `provider`'s synchronous-completion requirement) but that the
   transport layer copies tensor *values* under a freshly-named
   `TensorResourceId` on every node instead of passing an existing
   device-resident resource *reference* through when producer and consumer
   share a Provider/Device.

`device-resident-resource` and `device-memory-pool` are already normatively
specified (archived from earlier changes) -- this is not new architecture to
invent, it is existing Core contract the CUDA Provider
and the first-native dispatch loop do not yet honor. The Core P0s fixed in
`generalize-first-native-provider-dispatch` (dynamic Provider/Device
resolution, structured `TensorValue` write errors) were a prerequisite for
this work and are done; this is the next gate the intermediate audit already
predicted ("Gate 2").

## What Changes

- CUDA Provider (`providers/cuda`) gains a persistent, Provider-owned device
  allocation table keyed by `TensorResourceId`, replacing the current
  `Mutex<BTreeMap<TensorResourceId, HostTensor>>` host-resident storage.
  Device buffers persist across separate Kernel invocations instead of being
  allocated and freed per call.
- CUDA Provider's kernel methods stop performing implicit
  `clone_htod`/`clone_dtoh`; upload/download become explicit operations
  gated by Resource residency and Kernel compatibility, per the already-
  shipped "CUDA Provider Explicit Data Movement" requirement this change
  makes the implementation actually satisfy.
- CUDA Provider's `write_tensor_value`/`read_tensor_value` can now produce
  and consume real `TensorValue::Opaque` values backed by device pointers
  held privately by the Provider (never exposed through any public Runtime
  API, per `device-resident-resource`'s "Zero Copy Does Not Expose Native
  Address" requirement).
- `magnetar-runtime/src/first_native_runtime.rs`'s per-node dispatch loop
  stops unconditionally materializing every edge to `HostTensor`: when an
  edge's value is `TensorValue::Opaque` and the consuming node's operator
  never dereferences tensor bytes directly (`matmul`, `attention`, `silu`,
  `mul`/`residual-add`, `embedding`), the value is passed through by
  resource id without a host round-trip. `rmsnorm`/`rope` (which manipulate
  raw floats in Rust) and the genuine boundaries (weight binding,
  KV-history concatenation, final output extraction) are unaffected and
  keep materializing. **Scope note**: this eliminates the *consumer's*
  D2H+H2D for a chained edge; the *producer's own* output is still
  downloaded-then-reuploaded under a different id by the existing
  transport (a separate round-trip, left as documented follow-up --
  design.md Decision 7). A previously-undiscovered, unrelated blocker was
  also fixed here: every `KernelResource` this loop built hardcoded
  `KernelMemoryClass::Host`, which would reject every CUDA Kernel
  invocation (CUDA advertises `Device`) independent of residency --
  `resolved_kernel_memory_class` now derives it from the Prepared Plan's
  resolved Provider binding.
- `KernelErrorCode` (`magnetar-runtime/src/kernel.rs`) gains an
  out-of-device-memory category, closing the gap where
  `CudaErrorCode::OutOfDeviceMemory` and `ProviderExecutionErrorCode`'s OOM
  category currently have no matching Kernel-level home.
- `CudaProvider::health()`/`execution_api()` inconsistency (device found but
  kernel compile failed leaves `health() == Available` while
  `execution_api() == None`, with no Provider Health state naming that
  combination) gets a named, testable resolution.
- `.github/workflows/gpu-runner-smoke.yml` is corrected to no longer
  describe `providers/cuda` as an empty template, and is re-dispatched
  against current `main` to validate this change's real GPU-path behavior.

## Capabilities

### New Capabilities

(none -- `device-resident-resource`, `execution-stream`, and
`device-memory-pool` already exist as archived Core capabilities; this
change makes CUDA Provider and first-native dispatch conform to them rather
than introducing new normative concepts.)

### Modified Capabilities

- `cuda-provider`: "CUDA Provider Explicit Data Movement" and "CUDA Provider
  Memory Manager Integration" (from the not-yet-archived
  `implement-cuda-provider-baseline`) become actually-implemented instead of
  aspirational; "CUDA Provider Conformance Scope" is widened to require
  `provider-data-movement`, no longer deferring it. "CUDA Provider
  Synchronous Execution" is unchanged -- CUDA Provider's own kernel
  submission stays synchronous in this change (see design.md's Non-Goals);
  the existing `ProviderExecutionHandle`/`status`/`complete` contract
  already satisfies the `provider` capability's "Synchronous Provider Still
  Uses Completion Contract" requirement without any trait change.
- `kernel`: `KernelErrorCode` gains an out-of-device-memory category under
  the existing "Kernel Error Categories" requirement.

## Impact

- `providers/cuda` (submodule `Magnetar-provider-CUDA`): device allocation
  table, explicit upload/download, `Opaque` `TensorValue` production,
  health/execution_api reconciliation.
- `providers/cpu` (submodule `Magnetar-provider-CPU`): unaffected --
  Reference CPU already satisfies the completion contract via the existing
  `ProviderExecutionHandle`/`status`/`complete` methods, no trait change.
- `magnetar-runtime/src/first_native_runtime.rs`: per-node dispatch loop
  stops forcing host materialization for eligible-operator inputs already
  Provider-resident; `dispatch_qwen_graph_node` gains a `NodeInputValue`
  parameter type, `dispatch_reference_cpu_operator[_multi]` gains a
  `NodeInputResource` parameter type; `resolved_kernel_memory_class` fixes
  the `KernelMemoryClass::Host` hardcoding.
- `magnetar-runtime/src/kernel.rs`: `KernelErrorCode` new variant.
- `.github/workflows/gpu-runner-smoke.yml`: corrected description,
  re-dispatched.
- Depends on `generalize-first-native-provider-dispatch` (done) for dynamic
  Provider/Device resolution and the structured `TensorValue` error channel.
