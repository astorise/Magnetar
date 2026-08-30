# Define Device Resident Execution And Zero Copy Resource Contract

## Why

Magnetar now defines:

- Tensor Resources
- Runtime Memory Manager
- Resource Affinity
- explicit data movement
- Kernel preparation
- Prepared Execution Plans
- ExecutionStreams
- CompletionTokens
- ResourceReadiness
- asynchronous resource lifetime

The next requirement is to ensure that these abstractions actually permit
high-performance Device-resident inference.

Without an explicit residency and zero-copy contract, an implementation could
remain architecturally conformant while repeatedly moving:

```text
GPU -> host -> GPU
```

between Operators or decode steps.

That would defeat:

- KV-cache residency
- prepared execution
- asynchronous overlap
- continuous batching
- generated Kernel optimization
- Provider-native graph execution
- low-latency decode

Magnetar therefore needs a first-class contract describing where Tensor
Resources reside, when they can remain resident, when a view can share existing
storage, when the host may map a Resource without copying it, and when explicit
data movement is required.

The architectural rule is:

```text
Runtime / Memory Manager
    owns:
        logical resources
        placement policy
        residency policy
        aliasing policy
        lifetime
        movement authorization

Provider
    owns:
        native allocation realization
        native mapping
        cache/coherency mechanics
        peer-copy mechanics
        native memory handles

Device
    describes:
        memory capabilities
        addressability
        visibility
        peer-access characteristics
```

## What Changes

This change defines:

- ResourceResidency
- MemoryDomain
- ResidencySet
- ResourceView
- ResourceMapping
- host mapping
- mapped-resource lifetime
- zero-copy eligibility
- zero-copy semantics
- host-visible Device memory
- pinned host memory
- unified/managed memory representation
- coherent and non-coherent mappings
- explicit synchronization around mappings
- explicit data movement
- copy avoidance
- peer Device access
- peer Device transfer
- cross-Provider restrictions
- resource import/export boundary
- aliasing
- sub-resource views
- storage ownership
- immutable weight residency
- KV-cache Device residency
- intermediate Tensor residency
- final-output staging
- Prepared Execution Plan residency assumptions
- memory-pressure interaction
- observability
- conformance

## Core Principle

Magnetar SHALL prefer keeping inference resources resident on the execution
Device where policy, capacity, and compatibility permit it.

This is a preference and optimization goal.

It SHALL NOT override:

- correctness
- Resource Affinity
- Memory Manager policy
- Device capability
- Provider compatibility
- host-staging prohibition
- security policy

## Zero Copy Definition

For Magnetar, zero-copy means:

```text
a consumer accesses the same underlying logical storage
without a byte-for-byte transfer into a second allocation
performed solely for that consumer
```

Zero-copy does not necessarily mean:

- no synchronization
- no mapping operation
- no page-table work
- no cache maintenance
- no address translation
- no Provider bookkeeping
- identical native addresses across participants

A zero-copy path MAY still require explicit readiness and coherency operations.

## Zero Copy Is Not Assumed

A Resource SHALL NOT be treated as zero-copy accessible merely because:

- CPU and GPU share a physical memory architecture
- the Provider reports unified memory
- the allocation is host-visible
- two Devices belong to the same vendor
- two Providers run in the same process

Eligibility SHALL be established explicitly.

## Memory Domain

Magnetar SHALL represent logical memory domains.

A MemoryDomain describes a class of accessibility/residency.

Baseline conceptual classes MAY include:

```text
host
device-local
host-visible-device
shared
managed
external
```

The vocabulary SHALL remain extensible.

The logical class SHALL NOT expose native memory heap IDs, pointer values, or
vendor-specific allocation handles.

## Memory Domain Identity

A MemoryDomain SHOULD identify:

- Provider binding where required
- Device binding where required
- visibility class
- mapping capability
- coherence characteristics
- portability/lifetime scope

## Resource Residency

Tensor Resource SHALL expose or permit Runtime to query logical residency.

Conceptually:

```text
ResourceResidency
    resource
    memory_domain
    ProviderBinding
    DeviceBinding?
    visibility
    state
```

Residency SHALL describe where the authoritative Resource storage currently
exists or is valid.

## Authoritative Storage

A Tensor Resource SHOULD have a well-defined authoritative storage state.

If multiple physical copies exist, Runtime/Memory Manager SHALL track which
copies are current.

A stale replica SHALL NOT be treated as equivalent to authoritative data.

## Residency State

Suggested states include:

```text
resident
mapping-pending
transfer-pending
replicated
evicting
evicted
invalid
```

Implementations MAY refine the representation.

## Residency Set

A Resource MAY have a ResidencySet when multiple valid copies or shared
accessible representations exist.

For example:

```text
weights
    -> GPU0 resident
    -> GPU1 resident
```

or:

```text
shared allocation
    -> host visible
    -> integrated GPU visible
```

Validity and coherency SHALL be explicit.

## Device Resident Resource

A Device-resident Resource is one whose authoritative or current valid storage
is directly usable by a Kernel on the relevant Device without mandatory
host-staging copy.

Examples include:

- model weights
- KV-cache pages
- intermediate activations
- attention workspace
- logits before host extraction

## No Mandatory Host Round Trip

Execution of consecutive Operators on the same compatible Provider/Device SHALL
NOT require intermediate Tensor Resources to pass through host memory.

Example:

```text
MatMul
   |
   v
Device Tensor
   |
   v
RMSNorm
```

SHOULD remain Device-resident when possible.

## Device Resident Weights

Loaded Model Instance SHOULD be able to keep weights resident in compatible
Device memory across multiple inference requests.

Weights SHALL not need to be copied from host for every Kernel invocation.

## Weight Replication

Runtime MAY replicate immutable weights across Devices.

Each replica SHALL preserve:

- artifact identity
- dtype/layout
- Resource identity relationship
- Device affinity
- validity

Immutable replicated weights MAY be read concurrently.

## Intermediate Tensor Residency

Intermediate Tensors SHOULD remain on the Device selected by the Prepared
Execution Plan while downstream consumers are compatible.

A downstream Operator SHALL not trigger host staging merely because the Runtime
needs to describe the Tensor.

## KV Cache Residency

KV Cache SHOULD remain resident on the Device used for decode whenever
compatible with policy and capacity.

The normal decode path SHOULD resemble:

```text
Device weights
      +
Device KV
      +
Device intermediate tensors
          |
          v
      Device Kernels
          |
          v
      Device logits
```

rather than:

```text
Device
  -> host KV
  -> Device
  -> host intermediate
  -> Device
```

## Final Output

Host-visible final output MAY require explicit movement or mapping.

Only the data required by the API SHOULD need to become host-visible.

For token sampling, Runtime MAY choose whether logits remain Device-side or are
made host-visible according to the sampling implementation.

## Resource View

Magnetar SHALL support logical Resource Views where safe.

A ResourceView references existing Resource storage with transformed logical
metadata such as:

- offset
- shape
- strides
- layout view
- slicing
- sub-range

A ResourceView SHALL NOT imply a copy.

## View Storage Ownership

A ResourceView SHALL NOT own underlying storage independently from its parent
allocation.

Storage lifetime SHALL remain valid while any in-flight or live View requires
it.

## View Identity

A View SHALL have logical identity distinct from the underlying allocation.

Two Views MAY refer to overlapping regions.

Runtime/Memory Manager SHALL preserve aliasing information.

## View Bounds

A View SHALL be bounds-checked against underlying Resource storage.

Invalid offsets, extents, strides, or overflow SHALL be rejected.

## Non-Contiguous Views

A View MAY represent non-contiguous layout where supported.

Kernel compatibility SHALL determine whether a non-contiguous View can be
consumed directly.

Runtime SHALL NOT silently materialize a contiguous copy unless explicit
conversion/movement policy permits it.

## View Materialization

If a consumer requires a materialized layout, Runtime SHALL represent that as
an explicit operation.

Conceptually:

```text
ResourceView
    |
    v
explicit layout/materialization operation
    |
    v
new TensorResource
```

Materialization SHALL not be hidden as ordinary View creation.

## Aliasing

Runtime SHALL track aliasing relevant to correctness and asynchronous resource
lifetime.

Aliasing information SHALL participate in:

- read/write hazards
- memory reuse
- Prepared Plan validation
- ResourceReadiness
- mapping safety

## Zero Copy View

A ResourceView is naturally zero-copy if it refers directly to compatible
underlying storage without materialization.

Zero-copy View SHALL preserve underlying residency and affinity.

## Resource Mapping

A ResourceMapping represents temporary authorized access to Resource storage
from another logical access domain without necessarily copying it.

Typical case:

```text
Device Resource
     |
     v
host mapping
     |
     v
host-visible bytes
```

## Mapping Is Explicit

Host access to Device-resident Resource SHALL require explicit mapping,
movement, or host-visible storage semantics.

Runtime SHALL NOT expose raw Device pointers to ordinary host code.

## ResourceMapping Identity

ResourceMapping SHALL be an opaque logical object.

Conceptually:

```text
ResourceMapping
    id
    resource
    access
    mapped_domain
    range
    state
```

It SHALL NOT expose a native mapping handle through public Runtime contracts.

## Mapping Access

A mapping SHALL declare access mode.

Suggested modes:

```text
read
write
read-write
```

Access SHALL participate in synchronization and aliasing rules.

## Mapping Range

Runtime SHOULD permit mapping a bounded Resource region rather than requiring
the full allocation when Provider capability permits it.

Ranges SHALL be validated for overflow and bounds.

## Mapping Readiness

A read mapping SHALL not become usable until pending writes affecting the mapped
region are complete.

A write mapping SHALL be ordered against conflicting Device accesses.

## Mapping Lifetime

A ResourceMapping SHALL have explicit lifetime.

The mapped Resource SHALL not be evicted, destroyed, or incompatibly reused
while the mapping remains active.

## Mapping Release

Mapping release SHALL notify Runtime/Provider that mapped access has ended.

For write mappings, release MAY establish a new readiness/coherency transition.

## Mapping Does Not Grant Pointer Portability

An implementation MAY internally expose a native host pointer to trusted Runtime
code.

That pointer:

- SHALL remain mapping-scoped
- SHALL not become Resource identity
- SHALL not be serialized
- SHALL not cross Provider ABI except according to explicit buffer contract
- SHALL not be exposed to WASM Components as native memory authority

## Coherent Mapping

A coherent mapping MAY allow host/Device visibility without explicit cache
maintenance after synchronization.

The Provider SHALL accurately advertise this capability.

Synchronization requirements still apply.

## Non-Coherent Mapping

A non-coherent mapping SHALL require Provider-managed visibility operations.

Conceptually these may correspond to:

- flush
- invalidate
- synchronize
- cache maintenance

Magnetar public contracts SHOULD describe the semantic transition rather than
vendor-native operations.

## Host Visibility Transition

Before host reads mapped non-coherent Device-written data, Provider SHALL
perform required native visibility operations.

Before Device reads host-written mapped data, Provider SHALL perform required
visibility operations.

## Pinned Host Memory

Provider MAY support host allocations optimized for asynchronous Device access
or transfer.

This MAY conceptually correspond to pinned/page-locked host memory.

Magnetar SHOULD represent this through memory capability/class metadata rather
than CUDA-specific terminology in Core contracts.

## Pinned Host Memory Is Not Device Local

Pinned host memory SHALL not be treated as equivalent to high-bandwidth
Device-local memory.

Selection and Memory Manager policy MAY prefer Device-local storage for hot
compute even if Device can directly access pinned host memory.

## Shared Memory

A Provider MAY expose Resource storage directly visible to both host and Device.

This MAY support zero-copy execution.

Shared visibility SHALL still carry:

- compatibility
- synchronization
- coherence
- performance characteristics

## Managed Or Unified Memory

Providers MAY expose managed/unified memory capabilities.

Runtime SHALL model them as memory-domain/residency capabilities.

Core SHALL not assume one vendor's managed-memory behavior.

## Managed Residency

A managed Resource MAY migrate physically underneath Provider control.

Runtime SHALL retain logical Resource identity while Provider manages native
placement according to declared capability.

Where migration behavior materially affects scheduling or performance, Provider
SHOULD expose appropriate metadata/observability.

## Explicit Residency Preference

Runtime/Memory Manager MAY express a preferred residency.

Examples:

```text
prefer Device 0 local memory
prefer host-visible memory
preserve source affinity
```

The preference SHALL not become a raw native allocation instruction.

## Residency Requirement

Some operations MAY require hard residency/affinity.

Example:

```text
PreparedKernel K
requires Device 0 compatible storage
```

A hard requirement SHALL be validated before submission.

## Zero Copy Eligibility

Runtime SHALL determine whether a consumer can access a Resource without an
explicit byte-copy.

Eligibility SHOULD consider:

- Provider compatibility
- Device compatibility
- memory domain
- visibility
- dtype/layout
- alignment
- Resource Affinity
- access mode
- coherence requirements
- current readiness
- policy
- required feature set

## Zero Copy Decision

Zero-copy SHALL be a Runtime/Memory Manager decision informed by Provider
capability.

Provider SHALL NOT silently reinterpret incompatible Resource as directly
accessible.

## Zero Copy Failure

If zero-copy is unavailable, Runtime MAY:

- perform explicit permitted transfer
- perform explicit permitted materialization
- choose another Kernel
- choose another Provider/Device
- fail according to policy

It SHALL NOT silently stage through host when host staging is forbidden.

## Explicit Data Movement

Any byte-copy that changes storage residency SHALL be represented as explicit
data movement.

Examples:

```text
host -> Device
Device -> host
Device A -> Device B
Provider A -> Provider B
```

Existing placement and host-staging policy SHALL remain authoritative.

## No Hidden Host Staging

A Provider or Runtime optimization SHALL NOT secretly perform:

```text
Device A
    -> host temporary
    -> Device B
```

when the movement contract or host-staging policy forbids it.

If host staging is required, that requirement SHALL be visible to Runtime
policy.

## Transfer Identity

A movement operation SHOULD identify:

- source Resource
- destination memory domain
- destination Provider/Device
- transfer mode
- synchronization dependencies
- result Resource
- host-staging requirement

## Asynchronous Movement

Data movement MAY execute asynchronously via ExecutionStream.

It SHALL produce CompletionToken/ResourceReadiness semantics.

## Transfer Completion

Destination Resource SHALL not be readable until movement completion or
equivalent ordered dependency is established.

## Source Lifetime During Transfer

Source storage SHALL remain alive until asynchronous movement no longer
references it.

## Copy Elision

Runtime SHOULD elide a transfer when source Resource is already directly
accessible to the selected consumer and all requirements are satisfied.

Copy elision SHALL preserve semantic equivalence.

## Redundant Transfer Elimination

Prepared Execution Plan MAY detect redundant movements during Plan build.

Example:

```text
Resource already Device-resident
    +
next Kernel same Device
    ->
no copy node
```

## Zero Copy And Prepared Plans

Prepared Execution Plan MAY encode residency assumptions and zero-copy resource
binding paths.

Examples:

- weights remain Device-resident
- KV slots remain Device-resident
- intermediate stays on same Device
- host mapping only at final boundary

These assumptions SHALL be guarded.

## Residency Guard

A Prepared Execution Plan MAY define a residency guard.

If actual Resource residency no longer satisfies the Plan:

- alternate compatible resident copy MAY be used
- explicit movement MAY occur if planned/permitted
- Plan may request rebind/replan
- execution SHALL fail if no safe path exists

## Plan Does Not Own Physical Memory

Prepared Execution Plan SHALL not own native allocations.

Memory Manager retains physical-lifetime authority.

## Stable Resource Binding

Model weights or persistent buffers MAY be pre-bound logically to stable
Resource IDs/residency classes.

The Plan SHALL not contain raw Device addresses.

## Dynamic Resource Binding

Session-specific KV and invocation Tensors SHALL be bound at execution time.

Their residency SHALL satisfy Plan guards.

## Peer Device Access

Provider MAY advertise that Device A can directly access memory owned by
Device B.

Peer access SHALL be explicit capability.

Runtime SHALL NOT infer peer access merely from:

- same vendor
- same host
- same Provider
- same interconnect family

## Peer Access Modes

Provider capability MAY distinguish:

```text
peer-read
peer-write
peer-read-write
peer-copy
```

and additional constraints.

## Peer Access Performance

Direct peer access being technically possible SHALL not imply it is optimal.

Selection/Memory policy MAY prefer replication or transfer.

## Peer Device Zero Copy

If Device A can safely and efficiently consume Device B Resource directly,
Runtime MAY permit zero-copy peer access.

Resource Affinity, synchronization, and capability SHALL all permit it.

## Peer Transfer

Provider MAY perform direct Device-to-Device transfer where supported.

This SHALL remain an explicit movement operation even if host staging is
avoided.

## Cross-Provider Zero Copy

Baseline Magnetar SHALL NOT assume zero-copy Resource sharing across different
Providers.

A future explicit interoperability capability MAY enable it.

Without such capability, Runtime SHALL use explicit supported movement or reject
the path.

## Native External Memory Handles

Raw platform handles such as:

- DMA-BUF
- file descriptors
- NT handles
- IOSurface handles
- CUDA IPC handles
- Vulkan external-memory handles
- Metal native objects

SHALL NOT appear in general Runtime Resource contracts.

## Future Interop Capability

A future explicit Resource Interoperability Capability MAY define safe opaque
import/export semantics.

It SHALL be separate from ordinary Tensor Resource identity.

## Resource Import

Provider MAY import externally backed memory only through an explicit,
authorized capability.

Import SHALL validate:

- size
- alignment
- access rights
- lifetime
- Provider compatibility
- Device compatibility
- security policy

## Resource Export

Resource export SHALL be explicit and policy-controlled.

Runtime SHALL NOT automatically export native memory handles to Components,
CLI callers, or inference clients.

## WASM Boundary

WASM Components SHALL interact through logical Tensor Resources and portable
data-movement semantics.

They SHALL NOT receive:

- GPU pointers
- mapped host pointers
- DMA-BUF handles
- CUDA IPC handles
- native shared-memory handles

## Memory Pressure

Device residency SHALL remain subject to Memory Manager pressure policy.

Under pressure, Runtime MAY:

- evict
- spill
- replicate less
- rebuild Plan
- use another Device
- reject admission

according to explicit policy.

## Eviction

Eviction SHALL preserve Resource correctness.

A Resource with in-flight access SHALL not be physically evicted until safe.

## Eviction Destination

If eviction requires movement to another domain, that movement SHALL be
explicit and synchronization-safe.

## Spill To Host

Spilling Device Resource to host SHALL obey host-staging policy.

A policy forbidding host staging SHALL prevent such spill for affected
resources.

## Residency Pinning

Policy MAY pin a Resource to a residency domain.

Possible uses include:

- latency-critical weights
- active KV cache
- graph-capture resources

Pinning SHALL remain bounded by admission/capacity policy.

## KV Cache Under Pressure

KV-cache eviction/spill SHALL preserve:

- Session ownership
- page readiness
- sequence ordering
- active in-flight references
- host-staging policy

## Prefix Cache Residency

Prefix Cache entries MAY be Device-resident and shared read-only.

Reuse on another Device may require:

- compatible shared access
- replica
- explicit transfer

## Adapter Residency

Loaded adapter weights MAY remain resident and share normal persistent Resource
residency semantics.

## Quantized Resources

Residency and zero-copy SHALL preserve quantization metadata.

A consumer expecting different packing/layout SHALL not treat the same bytes as
compatible merely because they are resident.

## Alignment

Zero-copy eligibility SHALL account for Kernel/Provider alignment requirements.

An incompatible alignment MAY require:

- another Kernel
- explicit materialization
- transfer
- failure

## Layout Compatibility

Resident storage SHALL not bypass layout compatibility.

A Device-local Tensor with incompatible layout is still incompatible.

## Dtype Compatibility

Resident storage SHALL not bypass dtype compatibility.

No implicit reinterpretation is permitted.

## Read-Only Resources

Runtime MAY optimize sharing and replication of immutable read-only Resources.

Mutation SHALL require a distinct writable contract.

## Copy-On-Write

A future implementation MAY support copy-on-write Views.

If implemented, materialization on write SHALL be explicit in Resource state and
synchronization semantics.

Copy-on-write SHALL not create hidden semantic mutation.

## Error Model

Structured errors SHOULD include:

```text
resource-residency-unknown
resource-residency-invalid
resource-residency-incompatible
resource-residency-required
resource-residency-evicted
resource-residency-transfer-pending

resource-zero-copy-unavailable
resource-zero-copy-policy-denied
resource-zero-copy-provider-incompatible
resource-zero-copy-device-incompatible
resource-zero-copy-layout-incompatible
resource-zero-copy-alignment-incompatible
resource-zero-copy-coherency-unsupported

resource-view-invalid
resource-view-out-of-bounds
resource-view-overflow
resource-view-layout-unsupported
resource-view-materialization-required

resource-mapping-unsupported
resource-mapping-access-denied
resource-mapping-not-ready
resource-mapping-conflict
resource-mapping-range-invalid
resource-mapping-coherency-failed
resource-mapping-release-failed

resource-transfer-required
resource-transfer-denied
resource-transfer-host-staging-denied
resource-transfer-peer-unsupported
resource-transfer-failed

resource-peer-access-unsupported
resource-peer-access-denied

resource-eviction-denied
resource-eviction-in-flight
resource-spill-denied

internal-resource-residency-error
```

## Observability

Residency observability MAY include:

```text
resource-allocated
resource-resident
resource-replicated
resource-view-created
resource-mapped
resource-unmapped
resource-transfer-started
resource-transfer-completed
resource-transfer-elided
resource-peer-access-used
resource-eviction-started
resource-evicted
resource-spilled
resource-residency-guard-failed
zero-copy-selected
zero-copy-unavailable
```

Observability MAY report:

- ResourceId
- logical MemoryDomain
- Provider/Device stable identity
- byte size
- dtype/layout
- access mode
- mapping type
- transfer kind
- zero-copy decision reason
- residency transition
- pressure class

Observability SHALL NOT expose:

- native Device pointers
- mapped host addresses
- DMA-BUF handles
- file descriptors
- native external-memory handles
- model-weight contents
- KV contents
- prompt contents
- secrets
- credentials

## Conformance

Conformance SHALL validate:

- consecutive same-Device Operators do not require host round-trip
- Device-resident weights can be reused across requests
- KV cache can remain Device-resident across decode steps
- intermediate Resources can remain Device-resident
- zero-copy does not bypass dtype/layout/alignment compatibility
- zero-copy does not bypass Resource Affinity
- View creation does not copy
- View bounds/overflow are validated
- aliasing remains visible to synchronization/lifetime logic
- host mapping waits for Resource readiness
- mapping lifetime prevents unsafe eviction/reuse
- non-coherent mappings perform required semantic visibility transitions
- no native pointer becomes portable Resource identity
- explicit movement is used when residency changes
- host staging prohibition is enforced
- redundant transfer can be elided
- asynchronous transfer preserves source/destination lifetime
- peer access is capability-driven rather than assumed
- cross-Provider zero-copy is denied without explicit interop
- Device remains descriptive rather than allocation/synchronization authority
- Memory Manager remains residency/lifetime policy authority
- Prepared Plan does not own native memory
- eviction does not race in-flight execution
- observability redacts native memory state

## Non-Goals

This change does not:

- require zero-copy on every platform
- expose GPU pointers
- expose DMA-BUF or native memory handles
- define CUDA Unified Memory semantics directly
- define Vulkan external memory
- define IOSurface
- define RDMA
- define distributed memory
- define tensor-parallel collective communication
- define one universal allocator
- move Memory Manager authority into Provider
- allow hidden host staging
- make Device an allocation API
- require peer access
- define the future cross-Provider interop ABI

## Impact

Magnetar gains an execution resource model capable of preserving Device
residency across the inference graph:

```text
Model weights ─────────────┐
                           │
KV cache ──────────────────┤
                           ▼
                    Device-resident
                    execution graph
                           │
Intermediate tensors ──────┤
                           │
                           ▼
                    final output
                           │
                    explicit boundary
                           │
                    map / transfer
                           ▼
                          host
```

This allows the asynchronous execution model introduced by ExecutionStreams to
remain effective instead of being undermined by implicit host copies.