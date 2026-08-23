# Define Runtime Memory Manager

## Why

Magnetar is an inference Runtime.

Future inference features require explicit memory ownership, allocation,
residency, dtype placement, staging, and pressure management.

The current architecture already separates Compute, Provider, Device, Planning,
Scheduler, Resolution, and Resource Affinity.

However, memory is still not represented as a first-class Runtime subsystem.

This creates a coupling risk.

If memory behavior is hidden inside Compute, Device, Provider, or Planning,
Magnetar will struggle to implement advanced inference features such as:

- caching allocator
- asynchronous arena allocation
- pending allocation queues
- pinned host memory
- zero-copy feasibility checks
- host/device staging
- memory pressure
- storage dtype versus compute dtype
- quantized model residency
- KV cache residency
- prefix cache residency
- adapter residency
- batching memory admission
- model loading memory planning
- transfer feasibility
- explicit data movement
- browser-specific memory constraints

A TensorDescriptor is not enough.

A Device memory field is not enough.

A Compute plan is not enough.

Magnetar needs a Runtime-owned Memory Manager that coordinates memory
semantics across inference execution.

The Memory Manager SHALL be a first-class Runtime subsystem.

## What Changes

This change introduces the Runtime Memory Manager.

The Memory Manager owns memory-related Runtime responsibilities.

It SHALL be separate from:

- Compute contract definitions
- Device metadata
- Provider execution implementation
- Scheduler queueing
- Resolution Policy
- Component Runtime
- Model artifact identity

The target source ownership SHOULD include a memory module such as:

```text
magnetar-runtime/src/
├── memory.rs
```

or, if the implementation grows:

```text
magnetar-runtime/src/memory/
├── mod.rs
├── manager.rs
├── allocator.rs
├── arena.rs
├── allocation.rs
├── pending_queue.rs
├── pinned.rs
├── zero_copy.rs
├── staging.rs
├── dtype.rs
├── residency.rs
├── pressure.rs
└── error.rs
```

The exact file layout is implementation-defined.

The architectural requirement is that Memory Manager responsibilities are not
hidden inside Compute, Device, Provider, or Planning.

## Memory Manager Responsibilities

The Memory Manager SHALL own or coordinate:

- allocation requests
- allocation lifetime
- allocation handles
- allocation classes
- memory arenas
- caching allocator behavior
- pending allocation queues
- host memory
- pinned host memory
- device memory
- unified/shared memory where available
- zero-copy feasibility
- staging decisions
- transfer memory requirements
- storage dtype
- compute dtype
- tensor residency
- model artifact residency
- adapter residency
- KV cache residency
- prefix cache residency
- memory pressure reporting
- memory admission checks
- allocation diagnostics
- memory-related observability

## Relationship With Compute

Compute SHALL own portable tensor and operation descriptions.

Compute MAY define:

- TensorDescriptor
- TensorResourceDescriptor
- ComputeDType
- ShapeDescriptor
- LayoutDescriptor
- ComputeOperationRequest
- ComputeGraph
- DataMovementDescriptor

The Memory Manager SHALL own memory realization.

For example:

```text
TensorDescriptor
    describes shape, layout, dtype, semantic tensor properties

MemoryAllocation
    describes actual Runtime-owned memory reservation or allocation

TensorResidency
    describes where tensor content currently resides

MemoryPlacement
    describes host/device/shared/pinned/staged placement

MemoryManager
    validates and manages allocation, residency, and movement feasibility
```

Compute types SHALL NOT secretly own allocator state.

Compute validation MAY ask Memory Manager whether a request is feasible.

## Relationship With Device

Device metadata describes execution targets.

Device metadata MAY report memory capacity, memory type, features, and limits.

The Memory Manager consumes Device metadata.

Device SHALL NOT own global allocation policy.

Device SHALL NOT own Runtime-wide caching allocator behavior.

Device SHALL NOT own cross-Provider transfer staging policy.

## Relationship With Provider

Providers may allocate and own native resources.

Provider-owned resources SHALL be represented through Runtime-owned opaque
resource records and Resource Affinity.

The Memory Manager SHALL coordinate memory planning and memory pressure with
Providers.

Providers SHALL NOT expose raw native memory handles through portable APIs.

Provider-specific allocation handles may exist internally but SHALL remain
opaque to Components.

## Relationship With Planning

Planning SHALL consume Memory Manager decisions.

Planning may ask:

- is this allocation feasible?
- does this operation require staging?
- is zero-copy possible?
- is pinned memory required?
- is host staging allowed?
- is transfer feasible?
- does this batch fit?
- does this model residency fit?
- should this request wait in a pending queue?

Planning SHALL NOT own allocator internals.

## Relationship With Scheduler

Scheduler SHALL not allocate memory directly.

Scheduler may use Memory Manager admission results to decide whether to:

- admit work
- queue work
- delay work
- retry later
- fail with structured memory error
- trigger memory pressure policy

Memory pressure may influence scheduling.

## Caching Allocator

The Memory Manager SHALL define a caching allocator model.

The caching allocator MAY retain freed allocations for reuse.

It SHALL distinguish:

- reusable allocation
- active allocation
- pending allocation
- reserved arena memory
- provider-owned memory
- external memory
- pinned host memory
- device memory

The allocator SHALL be policy-governed.

It SHALL expose memory pressure and cache state for diagnostics.

## Asynchronous Arena

The Memory Manager SHALL support asynchronous allocation semantics.

Large allocations, device allocations, or Provider-mediated allocations may not
complete immediately.

The Memory Manager SHALL support pending allocation queues.

A pending allocation request SHALL have:

- request identity
- requested size
- memory class
- placement preference or requirement
- dtype requirements where relevant
- affinity requirements
- priority or admission metadata
- timeout/deadline where relevant
- cancellation state
- diagnostic reason

## Pending Queues

Pending queues SHALL be explicit.

A request that cannot be admitted immediately due to memory pressure or
allocator availability MAY be queued if policy allows.

The queue SHALL not hide failure forever.

A queued allocation must eventually:

- allocate
- time out
- be cancelled
- fail due to policy
- fail due to pressure
- fail due to Provider/Device status

Pending queues SHALL emit observability.

## Pinned Host Memory

Pinned host memory SHALL be explicitly represented.

Pinned memory may be needed for efficient host-device transfer.

The Memory Manager SHALL distinguish pinned memory from ordinary host memory.

Pinned memory SHALL be policy-limited because it can affect system performance.

A Component SHALL NOT request arbitrary pinned memory directly.

Pinned memory is Runtime-managed.

## Zero-Copy

Zero-copy SHALL be represented as a feasibility result, not an assumption.

The Memory Manager SHALL determine whether zero-copy is possible based on:

- source residency
- target residency
- Provider support
- Device support
- memory type
- alignment
- dtype/layout compatibility
- Resource Affinity
- host staging policy
- platform constraints
- browser constraints where applicable

If zero-copy is not feasible, Runtime policy determines whether staging,
copying, transfer, or failure occurs.

## Host Staging

Host staging SHALL remain explicit.

The Memory Manager SHALL respect HostStagingPolicy from Compute.

If host staging is forbidden, the Memory Manager SHALL not insert host staging
silently.

If host staging is permitted, the Memory Manager may still reject staging due to
policy, memory pressure, platform restrictions, or Provider constraints.

## Storage DType Versus Compute DType

The Memory Manager SHALL represent the distinction between:

```text
storage_dtype
compute_dtype
```

Examples:

```text
storage_dtype = int8
compute_dtype = bf16

storage_dtype = q4_k
compute_dtype = fp16

storage_dtype = fp16
compute_dtype = fp32
```

Storage dtype describes how data is stored in memory.

Compute dtype describes how operations execute.

The Memory Manager SHALL use this distinction for:

- allocation size
- transfer size
- staging requirements
- dequantization workspace
- temporary compute buffers
- model residency
- adapter residency
- KV cache layout
- batching admission

Compute operation semantics may still define supported compute dtypes.

Memory Manager owns memory consequences.

## Tensor Residency

The Memory Manager SHALL track tensor residency.

Residency MAY include:

- host ordinary memory
- host pinned memory
- device memory
- unified/shared memory
- provider-owned opaque memory
- external borrowed memory
- browser linear memory
- staged temporary memory

Residency SHALL be Runtime-owned state.

Portable Components SHALL not forge residency.

## Model Residency

Future model loading will require model artifact residency.

This change prepares the Runtime to represent model residency without defining
the full Model Artifact model.

Model residency MAY include:

- weights stored compressed
- weights stored quantized
- weights materialized for compute
- provider-resident weights
- device-resident weights
- sharded weights
- adapter overlays
- temporary dequantization buffers

The full Model Artifact contract is defined later.

## KV Cache And Prefix Cache

The Memory Manager SHALL be the future owner of KV cache and prefix cache memory
admission and residency.

This change does not implement the KV cache model.

It establishes that KV cache memory SHALL not be hidden inside Scheduler,
Provider, or Generation code.

## Memory Pressure

The Memory Manager SHALL report memory pressure.

Pressure may be scoped by:

- Runtime
- Provider
- Device
- memory class
- arena
- cache
- model residency
- KV cache
- temporary workspace

Pressure levels SHOULD align with Provider pressure concepts where useful:

```text
unknown
low
moderate
high
saturated
```

Memory pressure SHALL influence admission, planning, scheduling, and Provider
status.

## Allocation Classes

The Memory Manager SHALL distinguish allocation classes.

Initial classes SHOULD include:

- tensor
- model-artifact
- tokenizer-artifact
- adapter-artifact
- quantization-artifact
- kv-cache
- prefix-cache
- temporary-workspace
- transfer-staging
- host-pinned
- browser-linear-memory
- runtime-internal

The exact names may differ.

## Browser Target Considerations

The Memory Manager SHALL support platform-specific memory constraints.

For browser targets, memory may be constrained by:

- WebAssembly linear memory
- browser memory limits
- JavaScript ArrayBuffer behavior
- WebGPU buffer constraints where applicable
- absence of native pinned memory
- absence of native dynamic Provider loading

Browser-specific behavior SHALL be expressed as Memory Manager capabilities and
policy, not by pretending native memory features exist.

Detailed browser Component Engine handling is defined by a later platform
Component Engine change.

## Memory Errors

Memory errors SHALL be structured.

Error categories SHOULD include:

- allocation denied
- allocation pending
- allocation timeout
- allocation cancelled
- out of memory
- memory pressure saturated
- unsupported memory class
- unsupported placement
- unsupported dtype storage
- unsupported compute dtype
- zero-copy unavailable
- staging forbidden
- staging denied by policy
- pinned memory unavailable
- provider allocation failed
- device memory unavailable
- browser memory limit exceeded
- invalid allocation handle
- resource affinity conflict

## Observability

The Memory Manager SHOULD emit observations for:

- allocation requested
- allocation admitted
- allocation queued
- allocation completed
- allocation failed
- allocation released
- cache hit
- cache miss
- cache eviction
- arena growth
- arena pressure
- pending queue delay
- pinned memory usage
- zero-copy accepted
- zero-copy rejected
- staging inserted
- staging denied
- memory pressure change

Observability SHALL not control memory decisions.

## Non-Goals

This change does not:

- implement full model loading
- define full Model Artifact format
- define KV cache semantics
- define prefix cache semantics
- define batching scheduler
- define WebGPU Provider
- define browser Component Engine
- define Provider ABI memory structs
- define distributed memory
- define cross-node memory transfer
- define out-of-process Provider memory
- expose raw memory handles to Components
- allow Components to allocate arbitrary host/device memory
- allow Components to request pinned memory directly
- silently insert forbidden host staging

## Impact

Magnetar gains an explicit Memory Manager.

Future inference features will have a correct ownership point for allocation,
residency, dtype placement, staging, zero-copy, and pressure.

The architecture becomes:

```text
Compute
  describes tensors and operations
        |
        v
Planning
  asks memory feasibility
        |
        v
Memory Manager
  owns allocation, residency, staging, pressure
        |
        v
Provider / Device
  execute and expose native capabilities
```

This prepares the next inference-domain changes:

- model artifact model
- tokenizer contract
- generation contract
- inference session model
- KV cache model
- model loading contract
- sampling and logits processing contract