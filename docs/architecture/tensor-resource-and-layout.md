# Tensor Resource And Layout Contract

Operators describe computation. Kernels execute implementations. Memory
Manager owns allocation and residency. The Tensor Resource and Layout
contract is the precise, portable vocabulary both Operators and Kernels use
to describe tensor shape, dtype, layout, view, aliasing, memory class, and
Resource Affinity, so tensor handling stays unambiguous and safe.

## Where The Contract Lives

Rather than a single monolithic type, the contract is split across the
modules that already own the concepts it depends on:

- `compute` owns portable `TensorDescriptor`, `TensorResourceId`,
  `TensorResourceDescriptor`, `ShapeDescriptor`, `DTypeDescriptor`,
  `LayoutDescriptor`, and `ViewDescriptor` — the metadata that does not imply
  allocation.
- `memory` owns `TensorResidency`, `MemoryPlacement`, and the Memory
  Manager's tensor-residency table (allocation binding, placement, staged
  state).
- `affinity` owns `ResourceAffinity` — Runtime-derived, authoritative
  Provider/Device/Capability/Artifact binding that a caller cannot forge.
- `execution_graph` owns edge-level `TensorAliasing`, `TensorMutability`,
  and `TensorResidencyConstraint` for graph planning.
- `operator` owns `TensorLayoutKind` and `TensorRole` for Operator contracts.
- `tensor` (this module) owns what did not exist anywhere yet: lifecycle
  state, readiness distinct from lifecycle, Tensor-Resource-level mutability
  and aliasing (finer-grained than the graph-edge classification), a
  portable memory-class enum, `TensorResource` and `TensorView` as
  first-class values, structured `TensorError` categories, and
  `TensorObservation` events.

Rust type names are implementation-defined; this split is a design decision,
not a contract requirement. Consolidating everything into one module would
have meant migrating call sites across `kernel_dispatch`, `reference_cpu`,
`planning`, `memory`, `operator`, and `execution_graph` for no functional
gain, so the new module composes the existing types instead of duplicating
them.

## Resource Versus Memory Allocation, Provider-Owned Storage, And KV Cache

A `TensorResource` is not a `memory::MemoryAllocation`. `TensorResidency`
(embedded in `TensorResource`) *references* an optional
`MemoryAllocationId` when Memory Manager owns the backing bytes, but a
Provider-owned opaque resource has no Memory Manager allocation at all —
`residency.provider_owned` is `true` and `residency.allocation` stays
`None`. `KvCache` (see `kv_cache`) is a distinct allocation class and
lifecycle with its own paging and retention policy; a KV cache page may be
*described* through a `TensorResource` with `LayoutDescriptor::Paged` once
paged layout support lands, but the cache's own identity, retention, and
sharing rules stay owned by `kv_cache`, not by this contract.

## Descriptor Versus Resource

A `TensorDescriptor` (shape, dtype, layout, optional view) never implies
storage. A `TensorResource` binds a `TensorResourceId` to that descriptor
plus `TensorResidency` (placement, allocation, affinity), a `lifecycle`
state, and a `readiness` state. Creating a descriptor allocates nothing;
creating a `TensorResource` in the `Ready` lifecycle state means Memory
Manager has bound real storage to it.

## Lifecycle

`TensorLifecycleState` is Runtime- and Memory-Manager-owned:

```text
declared -> planned -> allocating -> ready -> in-use | view | mutating -> ready
ready -> released | evicted
evicted -> allocating (rehydration)
any -> invalid
declared | planned | allocating | mutating -> failed
```

`TensorResource::transition_to` rejects transitions outside this table.
`TensorResource::ensure_usable` rejects use of a released, invalid, or
failed resource, or one whose lifecycle is not yet dispatchable
(`Ready`/`InUse`).

## Readiness

Readiness is tracked separately from lifecycle: a `Ready`-lifecycle
resource can still be `PendingTransfer`, `PendingConversion`, or
`PendingCompute`. Kernels dispatch only against resources whose readiness is
`Ready`; `TensorReadiness::blocks_dispatch` is the compatibility check.

## Mutability And Aliasing

Execution Graph edges only need a coarse mutable/immutable classification
for planning. Tensor Resources need more: `TensorMutabilityKind` adds
`SingleWriter`, `MultiReader`, `RuntimeInternal`, and `ProviderOwned`, with
`allows_mutation()` deciding whether a Kernel-requested mutation is legal.
`TensorMutabilityKind` narrows to `execution_graph::TensorMutability` via
`From` for graph planning that only needs the coarse view.

`TensorAliasingKind` (`NoAlias`, `ReadOnlyAlias`, `MutableAlias`,
`InputOutputAlias`, `ViewAlias`, `InternalTemporaryAlias`) is validated
before dispatch by `validate_aliasing_for_dispatch`: an aliased
input/output pair is rejected unless the Kernel declares in-place support.
`validate_mutability_for_dispatch` rejects a mutation request against an
immutable resource.

Both validators return `TensorError::AliasingViolation` /
`TensorError::MutabilityViolation` rather than panicking or silently
allowing the operation.

## Memory Class

`TensorMemoryClass` (`Host`, `PinnedHost`, `Device`, `Unified`, `Shared`,
`ProviderOwned`, `BrowserLinearMemory`, `FutureWebgpuBuffer`) is a portable
classification derived from `memory::MemoryPlacement` via `From<&MemoryPlacement>`.
`validate_memory_class_for_kernel` rejects a resource whose class is outside
a Kernel's declared support set (an empty set means unconstrained).

## Shape And DType

`ShapeDescriptor` stays a plain `Vec<u64>` of concrete dimensions — every
call site (Reference CPU, planning, conformance, tests) already relies on
concrete extents for `element_count`/`byte_size`, so this contract does not
introduce symbolic or dynamic dimensions; that remains explicit future
work. What it does add is advisory dimension-role metadata: an optional
`Vec<DimensionRole>` (`Batch`, `Sequence`, `Hidden`, `Head`, `Other`) on
`TensorDescriptor`, validated to match the shape's rank when present.
`ShapeDescriptor::row_major_strides` gives the explicit dense stride order
Contiguous layout implies, so "row-major" is a computable fact, not just a
comment.

`DTypeDescriptor` already distinguishes a tensor's storage dtype from
whatever a Kernel computes with by way of `TensorDescriptor::storage_dtype`
and `::compute_dtype` (both optional — most descriptors have one dtype for
both). Silent conversion stays impossible: nothing in the crate mutates
`dtype` implicitly, and the only paths that change it are the explicit
`ComputeDataMovementKind::DTypeConversion` movement and Reference CPU's
`dtype_conversion` kernel. Accumulation dtype is a capability-advertisement
concern (`PrecisionSupport::accumulation_dtypes`), not a per-descriptor
field, since it describes what a Provider can accumulate into, not what one
tensor is. Index and mask tensors get their own `TensorRole` variants
(`Index`, `Mask`) alongside the existing `Input`/`Output`/`Storage`/
`Compute`/`Accumulation`, settable via `TensorDescriptor::semantic_role`.

## Layout

`compute::LayoutDescriptor` now covers every category the contract
requires: `Contiguous`, `Strided`, `Blocked`, `Paged`, `PackedQuantized`,
`AttentionSpecific`, `BrowserCompatible`, and `ProviderOpaque`. `Blocked`,
`Paged`, and `PackedQuantized` carry placeholder metadata (block
dimensions; page/block size and capacity; quantization method, bits per
value, group size, scale/zero-point dtype, packing order) and are not yet
implemented by any Provider — Reference CPU continues to advertise
`Contiguous` only, so
an unsupported layout fails with a structured error or requires an explicit
conversion plan, never a silent fallback. `LayoutDescriptor::kind()` maps
every variant to the matching `ComputeLayout` classification, and
`operator::layout_kind` maps it to the matching `TensorLayoutKind` used by
Operator contracts.

## Residency And Resource Affinity

`memory::TensorResidency` is the single record of where a tensor's bytes
actually live: `placement`, an optional `allocation` (Memory Manager owns
the allocation table), `provider_owned`/`staged` flags, and — now —
`eviction_eligible` and `size_bytes_estimate`. `TensorResidency::memory_class()`
derives the portable `TensorMemoryClass` from `placement`, and
`is_host_visible()` answers "can host code read this directly" from that
classification. Transfer/conversion state is not duplicated on
`TensorResidency`; it already lives on `TensorResource::readiness`
(`PendingTransfer`, `PendingConversion`) once a residency is wrapped in a
resource.

`ResourceAffinity` (in `affinity`) is what `TensorResidency.affinity` and
`TensorResourceDescriptor.affinity` both carry, and what
`TensorDescriptor::affinity_constraints` optionally declares up front.
`AffinityConstraints::merge` is the enforcement point: merging two
affinities that disagree on Provider or Device returns
`AffinityError::ProviderMismatch` / `DeviceMismatch` rather than silently
picking one — a caller cannot forge affinity by asserting a different
Device than the one Memory Manager actually bound. Kernel selection
consumes the same `ResourceAffinity` values, so a resource bound to a
Device stays bound through selection unless an explicit
`ComputeDataMovementDescriptor` (transfer/placement-conversion) moves it.

## Conversion

Every conversion the contract requires explicit — dtype, layout, memory
class movement, device transfer, host staging, opaque materialization,
quantization, dequantization — already has a home in
`compute::ComputeDataMovementKind` (`Upload`, `Download`, `Copy`,
`Materialize`, `Transfer`, `DTypeConversion`, `PlacementConversion`) plus
`HostStagingPolicy::Forbid`/`Permit`. Quantization and dequantization are
explicit Kernel-level operations rather than a distinct data-movement kind
— Reference CPU's `dequantize_placeholder` kernel is the first instance —
since "quantize this tensor" is closer to "run a kernel" than "move
bytes." Nothing in the crate performs any of these conversions as a side
effect of an unrelated call; they only happen where a caller explicitly
builds a `ComputeDataMovementDescriptor` or invokes a conversion Kernel.

## Runtime Tensor APIs And Access Boundary

`TensorResourceId` is an opaque, Runtime-issued string wrapper — nothing
about it encodes a pointer, handle, address, or path, and resolving an
unknown or forged id fails (`TensorError::ResourceNotFound`,
`MemoryError::InvalidAllocationHandle`) rather than granting access.
`TensorDescriptor`, `TensorResource`, `TensorResidency`, and `TensorView`
have no raw-pointer-shaped fields anywhere in their definitions, so the "no
raw pointers, native handles, allocation addresses, Provider/Device
internals, raw KV cache contents, raw model weights, or raw prompts"
guarantee is structural, not a runtime check. Where free-form diagnostic
text does flow through (`TensorError` reasons, `TensorObservation`
messages, Provider/Device diagnostics), it passes through
`compute::redact_backend_diagnostic` before being stored. Provider-owned
opaque storage is only ever reachable through a Runtime-created
`KernelInvocation` / `ProviderExecutionApi` call; Components see
descriptor-level metadata, never the Provider's native handle.

## Browser Compatibility

`TensorMemoryClass::BrowserLinearMemory` and `::FutureWebgpuBuffer`, plus
`ComputeLayout`/`LayoutDescriptor::BrowserCompatible`, give the browser
target explicit placement and layout categories without requiring
Wasmtime or native Provider loading — both stay behind
`not(target_arch = "wasm32")` feature gates elsewhere in the crate, and
`tensor` itself has no platform-specific code path. A browser feature this
contract does not yet implement (a specific WebGPU buffer layout, for
example) fails with `TensorError::BrowserFeatureUnsupported` rather than
silently falling back to a native path. `cargo check --target
wasm32-unknown-unknown` is part of this change's validation.

## Views

A `TensorView` borrows a base `TensorResourceId` plus a `ViewDescriptor`
(offset, strides), the view's own shape and dtype, its mutability and
aliasing, and its own `ResourceAffinity` (inherited from the base).
`TensorView::validate_against_base` rejects a view whose base has reached a
terminal lifecycle state (`Released`, `Invalid`, `Failed`) — a view SHALL
not outlive its base resource.

## Errors

`TensorError` gives every failure mode in the spec a structured, redacted
variant (descriptor/shape/rank/dtype/layout/memory-class/residency/
affinity/aliasing/mutability/view/size/materialization/transfer/browser-
feature categories, plus `Internal`). Constructors like
`TensorError::aliasing_violation` redact backend diagnostic text through
`compute::redact_backend_diagnostic` before it is stored, the same
redaction every other structured error type in this crate uses.

## Observability

`TensorObservation` pairs a `TensorObservationKind` (descriptor created,
resource planned/allocated/ready, view created, resource used/mutated,
conversion/transfer planned/completed/failed, released, evicted,
invalidated, aliasing violation, Resource Affinity conflict) with an
optional resource id and a redacted message. It never carries raw tensor
values, prompts, weights, cache contents, handles, or memory pointers.

## Non-Goals

This contract does not implement optimized layout transforms, full
quantized layout support, paged attention, or WebGPU tensors. It does not
expose raw tensor pointers or Provider-owned buffers to Components, and it
does not define a general ndarray library or training tensor gradients.
