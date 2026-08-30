# Define Device Memory Pool And Allocation Planner Contract

## Why

Magnetar now defines:

- Tensor Resources
- Runtime Memory Manager
- Resource Affinity
- Device residency
- zero-copy Resource access
- explicit data movement
- asynchronous ResourceReadiness
- CompletionTokens
- Prepared Execution Plans
- Device-resident weights
- Device-resident KV cache
- asynchronous workspace reuse

These contracts establish where inference data may reside and how its lifetime
remains safe.

They do not yet define how Device memory should be acquired efficiently.

A naive implementation could still perform:

```text
native allocate
Kernel execution
native free

native allocate
Kernel execution
native free

native allocate
Kernel execution
native free
```

for thousands of temporary Tensor Resources.

That would introduce:

- allocation latency
- Device allocator contention
- fragmentation
- synchronization around native frees
- unstable latency
- poor continuous-batching behavior
- KV-cache allocation churn
- inability to reserve predictable inference capacity

Magnetar therefore requires a first-class Device Memory Pool and Allocation
Planner model.

The architectural invariant is:

```text
Memory Manager decides:
    how much memory may be used
    what it is reserved for
    what may be evicted
    what may be reused
    what lifetimes may alias
    which logical pools exist

Provider realizes:
    native allocations
    native backing storage
    sub-allocation implementation details
    native memory registration

Device describes:
    capacity
    memory classes
    alignment/capability information
    pressure/status
```

## What Changes

This change defines:

- DeviceMemoryPool
- DeviceMemoryPoolId
- MemoryPoolClass
- MemoryPoolReservation
- PoolCapacity
- high/low watermarks
- hard and soft reservations
- AllocationClass
- AllocationRequest
- AllocationLease
- AllocationBlock
- sub-allocation
- Arena
- slab/bucket-style implementation freedom
- AllocationPlan
- AllocationPlanId
- allocation-plan generation
- Tensor lifetime intervals
- storage reuse
- alias-safe reuse
- alignment planning
- workspace planning
- persistent versus transient memory
- KV page pools
- batch workspace pools
- model-weight pools
- fragmentation metrics
- compaction policy
- relocation constraints
- eviction integration
- pressure response
- admission integration
- asynchronous reclamation
- allocation fallback
- OOM classification
- Prepared Execution Plan integration
- observability
- conformance

## Core Principle

The Runtime SHALL distinguish logical allocation policy from native allocation
implementation.

```text
Memory Manager
    |
    v
logical pool / allocation plan
    |
    v
Provider
    |
    v
native allocation/sub-allocation
```

A Device SHALL NOT become a general allocator API.

## Device Memory Pool

A DeviceMemoryPool represents a logical Runtime-managed capacity domain backed
by Provider-realized memory.

Conceptually:

```text
DeviceMemoryPool
    id
    class
    ProviderBinding
    DeviceBinding
    memory_domain
    capacity
    reservation
    watermarks
    policy
    state
```

A pool SHALL NOT expose native allocation handles or base pointers.

## Pool Identity

DeviceMemoryPoolId SHALL be opaque and Runtime-owned.

Pool identity SHALL distinguish materially different allocation policies or
memory domains.

It SHALL NOT be interpreted as:

- native heap ID
- CUDA memory pool pointer
- Vulkan memory heap
- allocator address
- Provider handle

## Pool Class

Pools SHOULD have logical classes.

Baseline conceptual classes MAY include:

```text
weights
kv-cache
workspace
transient
persistent
transfer
shared
```

The vocabulary SHALL remain extensible.

Pool class expresses Runtime policy intent.

It SHALL NOT prescribe one native allocator implementation.

## Pool Backing

A Provider MAY realize a logical pool using:

- one large native allocation
- multiple native allocations
- native memory pool API
- slabs
- arenas
- buddy allocator
- segregated free lists
- Device-specific allocator
- another strategy

The strategy remains Provider-private unless its characteristics affect Runtime
policy.

## Pool Capacity

A pool SHALL expose logical capacity accounting.

Conceptually:

```text
PoolCapacity
    configured_limit
    reserved_bytes
    committed_bytes
    leased_bytes
    reclaimable_bytes
    pending_reclaim_bytes
```

Exact representation MAY differ.

## Physical Capacity

Pool configured capacity SHALL not exceed compatible Device/runtime policy
unless overcommit is explicitly supported.

Provider-reported physical capacity MAY inform pool sizing.

## Hard Reservation

A hard reservation reserves capacity for a workload class that SHALL not be
silently consumed by lower-priority classes.

Example:

```text
KV cache hard reservation = 8 GiB
```

may protect active decode capacity.

## Soft Reservation

A soft reservation represents preferred capacity that MAY be borrowed according
to policy.

Borrowing SHALL remain visible to Memory Manager accounting.

## Reservation Scope

Reservations MAY be scoped to:

- Model Instance
- pool class
- tenant/deployment
- inference priority
- Device
- workload class

This change does not define multi-tenant security semantics.

## High Watermark

A pool MAY define a high watermark.

Crossing the high watermark MAY trigger:

- eviction
- cache trimming
- admission throttling
- reduced replication
- spill policy
- allocation-plan reconsideration

## Low Watermark

A low watermark MAY define the target pressure level after reclamation.

Example:

```text
high watermark = 90%
low watermark  = 75%
```

When pressure exceeds 90%, Runtime may reclaim until usage returns toward 75%.

## Critical Watermark

Policy MAY define a critical watermark near exhaustion.

Critical pressure MAY trigger stronger actions such as:

- rejecting new Model Instance load
- rejecting new Sessions
- cancelling optional autotuning
- disabling optional caches
- emergency eviction of reclaimable Resources

Hard safety and active in-flight lifetime SHALL still be preserved.

## Pool State

Suggested states are:

```text
initializing
ready
pressure
critical
draining
failed
closed
```

## Allocation Request

Memory Manager SHALL represent logical allocation requirements before Provider
native allocation.

Conceptually:

```text
AllocationRequest
    bytes
    alignment
    allocation_class
    memory_domain
    lifetime_class
    residency_requirement
    mutability
    aliasing_policy
    reclaimability
```

## Allocation Class

AllocationClass SHOULD distinguish behaviorally relevant memory uses.

Examples:

```text
model-weight
adapter-weight
kv-page
persistent-cache
execution-workspace
intermediate
transfer-staging
output
```

Allocation class SHALL remain extensible.

## Lifetime Class

Runtime SHOULD distinguish lifetimes such as:

```text
model-instance
session
execution-plan
batch-step
operator
temporary
cache-entry
```

Lifetime class helps planning and reuse but SHALL not replace actual lifetime
tracking.

## Allocation Lease

A successful logical allocation SHOULD produce an AllocationLease or equivalent
Resource backing relationship.

Conceptually:

```text
AllocationLease
    pool
    block
    offset
    length
    alignment
    generation
    state
```

The lease SHALL not expose a native pointer.

## Allocation Block

AllocationBlock represents Provider-backed memory capacity available for
sub-allocation.

A block MAY correspond internally to one or more native memory objects.

Runtime public contracts SHALL not expose those objects.

## Sub-Allocation

Multiple Tensor Resources MAY be backed by non-overlapping regions of one
AllocationBlock.

Example:

```text
AllocationBlock 256 MiB
|
+-- Tensor A  0..16 MiB
+-- Tensor B 16..48 MiB
+-- Workspace 48..80 MiB
+-- free ...
```

Sub-allocation SHALL preserve:

- bounds
- alignment
- lifetime
- ResourceReadiness
- aliasing
- residency
- Provider compatibility

## Arena

An Arena is a logical allocation region optimized for a compatible set of
lifetimes.

A Provider or Memory Manager implementation MAY use arenas.

The architecture SHALL NOT mandate one allocator algorithm.

## Persistent Memory

Persistent allocations MAY include:

- model weights
- long-lived adapter weights
- active Model Instance metadata buffers
- persistent Provider-prepared constants

Persistent allocations SHOULD avoid unnecessary churn.

## Transient Memory

Transient allocations MAY include:

- intermediate activations
- temporary conversion buffers
- per-operation workspaces
- short-lived outputs

Transient allocations SHOULD be eligible for aggressive reuse where lifetime
analysis permits.

## Workspace Memory

Kernel workspace requirements SHALL participate in allocation planning.

A workspace SHALL be reusable across non-overlapping executions when
CompletionToken/lifetime semantics prove safety.

## Workspace Upper Bound

Prepared Kernel metadata SHOULD expose workspace requirements or bounded
workspace estimation where applicable.

Prepared Execution Plan may use this information to reserve sufficient
workspace.

## Workspace Classes

A Plan MAY use one or more workspace classes rather than allocating distinct
storage per Kernel.

Example:

```text
workspace:small
workspace:attention
workspace:matmul-large
```

The actual strategy is implementation-defined.

## Allocation Planner

The Runtime Memory Manager SHALL support an Allocation Planner capable of
planning storage for a Prepared Execution Plan or related execution scope.

Conceptually:

```text
Execution Graph
     |
     v
logical Tensor lifetimes
     |
     v
workspace requirements
     |
     v
residency constraints
     |
     v
Allocation Planner
     |
     v
AllocationPlan
```

## Allocation Plan

An AllocationPlan describes how logical Resource storage requirements can be
satisfied from one or more pools.

Conceptually:

```text
AllocationPlan
    id
    generation
    scope
    pool_bindings
    slots
    lifetime_intervals
    reuse_groups
    reservation_requirements
    guards
```

## Allocation Plan Identity

AllocationPlan identity SHOULD incorporate materially relevant inputs such as:

- Execution Graph fingerprint
- Prepared Execution Plan scope
- dtype/layout
- shape envelope
- batch/sequence envelope
- Kernel workspace requirements
- memory-domain requirements
- pool policy version
- allocation policy version

It SHALL NOT include native addresses.

## Allocation Slot

An AllocationSlot represents a planned backing region for one or more
non-overlapping logical Resources.

Conceptually:

```text
AllocationSlot
    pool_class
    minimum_bytes
    alignment
    lifetime
    reuse_group
```

At execution time the slot is bound to actual AllocationLease-backed storage.

## Lifetime Interval

Planner SHOULD derive conservative lifetime intervals for intermediate
Resources.

Example:

```text
Tensor A: node 1 -> node 4
Tensor B: node 3 -> node 7
Tensor C: node 8 -> node 10
```

A and C may potentially reuse storage if no asynchronous overlap invalidates
the assumption.

## Asynchronous Lifetime

ExecutionStream concurrency SHALL participate in lifetime analysis.

Graph node ordering alone SHALL NOT be used to reuse storage if asynchronous
execution permits overlapping access.

CompletionToken semantics remain authoritative.

## Reuse Group

Planner MAY group Resources that can share backing storage at different times.

All members SHALL satisfy compatible:

- size
- alignment
- memory domain
- access
- residency
- lifetime
- synchronization constraints

## Reuse Is Not Aliasing Semantics

Planner-induced temporal storage reuse SHALL remain distinct from semantic
Tensor aliasing.

Two independent Tensors reusing the same bytes at different times are not
semantically aliases while both are live.

## Reuse Guard

A planned reuse SHALL occur only when previous use has reached required
completion.

Prepared Plan execution SHALL integrate this with ResourceReadiness.

## Alignment

Allocation planning SHALL respect Provider/Kernel alignment requirements.

Alignment arithmetic SHALL be checked for overflow.

## Alignment Classes

Memory Manager MAY bucket common alignments for efficient allocation.

A larger compatible alignment MAY satisfy a smaller requirement.

## Size Classes

Implementation MAY use size classes/buckets.

Size-class implementation SHALL not alter Tensor logical byte size.

Internal padding SHALL remain allocator metadata.

## Internal Fragmentation

Pool accounting SHOULD be able to distinguish logical requested bytes from
reserved/committed bytes where useful.

This permits observing internal fragmentation.

## External Fragmentation

Pool SHOULD be able to report inability to satisfy a large allocation despite
sufficient aggregate free bytes where fragmentation is the cause.

## Fragmentation Metrics

Metrics MAY include:

```text
free bytes
largest free region
requested bytes
committed bytes
internal fragmentation estimate
external fragmentation estimate
```

Exact algorithm is implementation-defined.

## Fragmentation Does Not Permit Unsafe Relocation

Runtime SHALL not move in-flight Resources merely to compact memory.

Any relocation SHALL preserve:

- completion
- mapping
- Views
- aliasing
- Provider preparation assumptions
- Prepared Plan residency assumptions

## Compaction

Memory Manager MAY support compaction where Provider/memory-domain capability
permits safe relocation.

Compaction SHALL be explicit Runtime-controlled memory management work.

It SHALL NOT occur as an invisible pointer rewrite unknown to Runtime logical
Resource state.

## Non-Movable Resources

Resources MAY be marked temporarily or permanently non-movable.

Examples:

- in-flight Resource
- active host mapping
- Provider graph-captured fixed-address Resource
- external imported Resource
- pinned residency Resource

## Movability

Resource movability SHALL be explicit.

A Resource SHALL not be assumed movable solely because it is a Tensor.

## Relocation

Safe relocation SHALL be represented as an explicit internal movement with new
backing storage and synchronization.

Logical Tensor Resource identity MAY remain stable if the Runtime contract
supports rebinding.

Provider-native bindings depending on old address SHALL be revalidated or
reprepared.

## Prepared Kernel Address Assumptions

A Prepared Kernel SHOULD NOT require stable Resource native addresses unless its
contract explicitly states such requirement.

If address stability is required, Memory Manager SHALL treat affected Resource
as pinned/non-movable for the relevant lifetime.

## Prepared Segment Address Assumptions

Provider-prepared graph/segment MAY have stronger address-stability
requirements.

Provider SHALL advertise those requirements.

Plan construction SHALL incorporate them into AllocationPlan.

## Allocation Plan And Prepared Execution Plan

PreparedExecutionPlan MAY reference an AllocationPlan generation.

Conceptually:

```text
PreparedExecutionPlan
    |
    +-- Kernel bindings
    +-- stream/dependency bindings
    +-- AllocationPlan
```

A hard-incompatible memory-plan change SHALL invalidate or require rebuilding
the Prepared Execution Plan.

## Stable Slots

Persistent Resource slots MAY be allocated during Model Instance load.

Examples:

- weights
- adapter weights
- static workspace
- graph-capture buffers

## Dynamic Slots

Dynamic slots MAY be leased per:

- Session
- batch
- invocation
- decode step

Dynamic slots SHALL remain bounded by pool capacity/admission.

## Pool Reservation During Plan Preparation

Plan preparation MAY reserve memory capacity before the Plan becomes ready.

This prevents declaring a Plan READY when required Device memory cannot
reasonably be provided.

## Reservation Versus Commitment

Reservation SHALL remain distinct from physical commitment where the Provider
supports lazy backing.

Runtime SHALL still account for promised capacity.

## Overcommit

Memory overcommit MAY be supported only through explicit policy.

If enabled, Runtime SHALL know:

- which pool may overcommit
- maximum ratio/budget
- reclaim/failure strategy

Overcommit SHALL NOT be implicit.

## Admission

Runtime admission SHOULD consider pool reservations and projected memory demand.

A request SHOULD be rejectable before partial execution when capacity is
predictably insufficient.

## Model Instance Admission

Loading a Model Instance SHOULD consider:

- persistent weight memory
- mandatory workspace
- minimum KV capacity
- pinned allocations
- Provider-prepared graph memory

## Session Admission

Session admission MAY consider expected KV growth.

Runtime MAY reserve initial or maximum KV capacity according to policy.

## KV Page Pool

KV cache SHOULD be able to use a dedicated or logically distinguished pool of
fixed/compatible page allocations.

Conceptually:

```text
KVPagePool
    page_size
    total_pages
    free_pages
    leased_pages
    pending_reclaim_pages
```

## KV Page Size

KV page size SHALL derive from Runtime KV-cache format requirements.

It SHALL not be dictated solely by the native allocator.

## KV Page Lease

A Session/sequence MAY lease KV pages.

Page lifetime SHALL remain completion-aware.

## KV Page Recycling

A page SHALL return to the free pool only after:

- Session/page ownership ended
- no in-flight execution references it
- no Prefix Cache or shared user retains it
- required readiness/completion has reached terminal-safe state

## KV Growth

Runtime MAY grow a Session's KV allocation incrementally.

Failure to acquire additional pages SHALL use explicit policy:

- admission/backpressure
- spill where permitted
- eviction of reclaimable cache
- alternate Device
- generation failure

## KV Fragmentation

Fixed page pools SHOULD reduce fragmentation for KV workloads.

This is an implementation goal, not a requirement to use one allocator
algorithm.

## Continuous Batching

Continuous batching SHOULD allocate batch and sequence workspaces from reusable
pool-backed slots.

Batch slot lifecycle SHALL remain CompletionToken-aware.

## Batch Workspace Pool

Runtime MAY maintain separate batch workspace capacity to prevent transient
batch spikes from consuming protected KV/weight capacity.

## Memory Class Isolation

Policy MAY isolate pool classes.

For example:

```text
weights     cannot consume reserved KV capacity
autotuning  cannot consume latency-critical decode reservation
```

Borrowing may be allowed only explicitly.

## Pool Borrowing

A pool MAY borrow unused soft-reserved capacity from another pool.

Borrowing SHALL be tracked.

Borrowed capacity SHOULD be reclaimable according to policy when the owning
class needs it.

## Reclaimable Allocation

Resources SHALL indicate whether backing storage is reclaimable.

Examples:

```text
reclaimable:
    Prefix Cache
    stale tuning workspace
    inactive optional replicas

non-reclaimable while active:
    model weights required by active Plan
    active KV
    in-flight workspace
```

## Reclamation

Memory pressure MAY trigger reclamation of eligible allocations.

Reclamation SHALL honor:

- in-flight completion
- pinning
- mapping
- aliasing
- cache ownership
- policy priority

## Pending Reclaim

A Resource selected for reclamation but still referenced asynchronously MAY be
counted as pending reclaim.

It SHALL not be counted as immediately free.

## Asynchronous Free

Native freeing MAY be asynchronous or deferred.

Memory Manager accounting SHALL distinguish:

```text
logically released
pending native reclaim
physically reusable
```

## Provider Native Memory Pool

Provider MAY internally use native pool facilities such as vendor Device memory
pools.

Core SHALL not depend on any vendor-specific API.

## Provider Pool Capability

Provider MAY advertise capabilities relevant to pool realization:

- large block allocation
- sub-allocation support
- asynchronous native free
- address stability
- movable allocations
- host-visible pools
- shared memory
- pool growth/shrink capability
- minimum alignment
- preferred allocation granularity

These remain metadata/capability inputs.

## Device Capacity

Device MAY expose:

- total compatible memory
- currently available estimate
- pressure level
- memory-domain capacities

Device SHALL not allocate directly.

## Pool Growth

Memory Manager MAY grow pool backing when policy permits and Device capacity is
available.

Growth MAY allocate additional Provider-backed blocks.

## Pool Shrink

Memory Manager MAY shrink unused pool backing.

Shrink SHALL not release blocks containing live or pending-reclaim allocations.

## Pool Drain

A pool MAY enter draining state.

Draining means:

- no new normal leases
- existing leases remain valid
- reclaim occurs as lifetimes end
- pool closes when safe

## OOM Categories

Runtime SHOULD distinguish memory failures.

Suggested classes include:

```text
pool-capacity-exceeded
device-capacity-exceeded
hard-reservation-conflict
fragmentation
alignment-unsatisfied
pinned-capacity-exhausted
kv-page-exhausted
workspace-capacity-exceeded
provider-allocation-failed
```

All SHALL map to structured Runtime errors.

## OOM Retry

Runtime MAY retry allocation after bounded reclamation.

Retry SHALL not form an unbounded allocation/reclaim loop.

## OOM Fallback

Policy MAY respond to allocation failure by:

- reclaiming caches
- reducing optional replicas
- selecting lower-workspace Kernel
- using alternate Prepared Plan
- moving workload to another compatible Device
- spilling where permitted
- rejecting admission

Fallback SHALL remain explicit.

## Kernel Selection Integration

Memory feasibility remains a hard eligibility input.

A Kernel requiring workspace unavailable under current pool policy SHALL not
win because of better latency.

## Autotuning Integration

Autotuning candidates SHALL use bounded tuning memory budgets.

Autotuning SHALL not consume protected inference pool capacity beyond explicit
policy.

## Performance Feedback

Memory pressure, fragmentation, and allocation delay MAY contribute to Kernel
Performance Model context.

They SHALL not redefine correctness.

## Allocation Latency

Runtime MAY observe allocation latency.

Repeated high allocation latency MAY motivate:

- pool growth
- allocation-plan reuse
- different size classes
- pre-reservation

The adaptive system SHALL not rewrite allocator code automatically.

## Allocation Plan Cache

Runtime MAY cache AllocationPlans.

The cache SHALL be distinct from:

- Kernel Artifact Cache
- Autotuning Cache
- Prepared Execution Plan Cache
- KV Cache
- Prefix Cache

## Allocation Plan Cache Key

A key SHOULD include:

- graph fingerprint
- workload/shape envelope
- pool policy version
- Provider/Device compatibility
- dtype/layout
- workspace requirements
- memory-domain constraints
- batching mode
- KV mode

## Cached Plan Revalidation

A cached AllocationPlan SHALL be revalidated against:

- current pool state
- capacity
- reservations
- Provider/Device compatibility
- required alignment
- current Kernel workspace requirements

## Plan Staleness

An AllocationPlan MAY become stale after:

- Kernel specialization changes
- workspace requirement changes
- pool policy changes
- memory pressure regime changes
- Model Instance revision
- Device capability change

## Plan Hard Invalidation

A Plan SHALL be invalid when required allocation semantics cannot be satisfied.

Examples:

- required pool removed
- alignment no longer supportable
- hard reservation policy changed incompatibly
- fixed-address requirement conflicts
- Device lost

## Security

Allocation metadata SHALL not expose model contents or native addresses.

Resource size metadata MAY itself be sensitive operational information and
SHOULD follow observability policy.

## Native Pointer Privacy

Pools, AllocationBlocks, AllocationLeases, and AllocationPlans SHALL NOT expose
native addresses through public Runtime contracts.

## WIT Boundary

WASM Components SHALL not control:

- pool creation
- allocator strategy
- native block size
- Device memory pointer
- native pool handle
- compaction

Components continue to request portable Tensor operations.

## Runtime Inference API

Normal inference callers SHALL NOT select native memory pool or allocator
algorithm.

High-level deployment/session memory policy MAY exist outside ordinary
generation requests.

## Error Model

Structured errors SHOULD include:

```text
memory-pool-not-found
memory-pool-not-ready
memory-pool-draining
memory-pool-failed
memory-pool-capacity-exceeded
memory-pool-critical-pressure
memory-pool-reservation-conflict
memory-pool-borrow-denied

memory-allocation-invalid
memory-allocation-size-overflow
memory-allocation-alignment-invalid
memory-allocation-alignment-unsatisfied
memory-allocation-fragmented
memory-allocation-provider-failed
memory-allocation-no-compatible-pool
memory-allocation-overcommit-denied

memory-allocation-plan-invalid
memory-allocation-plan-stale
memory-allocation-plan-incompatible
memory-allocation-plan-build-failed
memory-allocation-plan-capacity-insufficient

memory-allocation-lease-invalid
memory-allocation-lease-in-use
memory-allocation-release-pending

memory-compaction-denied
memory-compaction-resource-pinned
memory-compaction-resource-in-flight
memory-relocation-failed

memory-kv-page-exhausted
memory-workspace-exhausted

memory-reclamation-failed
memory-reclamation-insufficient
memory-oom-retry-exhausted

internal-memory-pool-error
```

## Observability

Memory pool observability MAY include:

```text
memory-pool-created
memory-pool-grown
memory-pool-shrunk
memory-pool-pressure
memory-pool-critical
memory-pool-draining

allocation-requested
allocation-leased
allocation-reused
allocation-released
allocation-reclaim-pending

allocation-plan-built
allocation-plan-cache-hit
allocation-plan-stale

fragmentation-detected
reclamation-started
reclamation-completed
compaction-started
compaction-completed

kv-page-leased
kv-page-reclaimed

oom-detected
oom-fallback-selected
```

Observability MAY report:

- logical pool ID
- pool class
- Device/Provider stable identity
- capacity
- committed bytes
- leased bytes
- reclaimable bytes
- pending reclaim bytes
- high/low watermark
- allocation class
- logical byte size
- fragmentation summaries
- reservation summaries
- allocation latency

Observability SHALL NOT report:

- native base pointers
- native allocation handles
- raw Tensor contents
- weights
- KV contents
- prompts
- secrets
- credentials

## Conformance

Conformance SHALL validate:

- Memory Manager owns pool policy
- Provider owns native realization
- Device exposes no allocation API
- pools contain no native pointer semantics
- same allocation plan can reuse storage safely
- asynchronous execution prevents premature reuse
- alignment requirements are enforced
- Tensor lifetimes may enable temporal reuse
- semantic aliases are distinct from temporal storage reuse
- hard reservations cannot be silently consumed
- soft borrowing remains policy-controlled
- high watermark can trigger reclamation
- reclamation does not free in-flight storage
- pending reclaim is not counted as immediately reusable
- fragmentation can produce distinct failure from total capacity exhaustion
- compaction does not move pinned/in-flight Resources
- Prepared graph address-stability constraints are respected
- Plan readiness may depend on memory reservation
- KV pages recycle only after safe completion
- batch workspace cannot steal protected KV capacity without policy
- OOM fallback cannot bypass Memory Manager policy
- cached AllocationPlan is revalidated
- native handles remain redacted

## Non-Goals

This change does not:

- mandate buddy allocation
- mandate slab allocation
- mandate CUDA memory pools
- define exact allocator data structures
- expose native allocation pointers
- move allocation policy into Device
- let Components choose allocator strategy
- define OS virtual-memory paging
- define RDMA memory registration
- define distributed memory pools
- define multi-host unified memory
- guarantee zero fragmentation
- guarantee zero allocation latency
- require compaction
- require overcommit

## Impact

Magnetar gains a Device memory management layer suited to sustained inference:

```text
Device capacity
      |
      v
Runtime Memory Manager
      |
      +-- weights pool
      +-- KV pool
      +-- workspace pool
      +-- transient pool
      |
      v
Allocation Planner
      |
      v
Prepared Execution Plan
      |
      v
reusable allocation slots
      |
      v
Provider native memory
```

The hot execution path can therefore reuse already planned Device-resident
storage rather than repeatedly invoking expensive native allocation/free
operations.