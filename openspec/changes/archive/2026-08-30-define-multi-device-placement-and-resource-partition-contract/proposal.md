# Define Multi Device Placement And Resource Partition Contract

## Why

Magnetar now defines:

- Tensor Resources
- Resource Affinity
- Device residency
- explicit data movement
- zero-copy eligibility
- peer Device access
- Device Memory Pools
- Allocation Plans
- Prepared Execution Plans
- ExecutionStreams
- CompletionTokens
- ResourceReadiness
- Kernel selection and specialization

These contracts are sufficient for efficient execution on one Device.

The next architectural requirement is allowing one Model Instance and one
Prepared Execution Plan to use multiple Devices in the same Runtime and host.

This is required for models that:

- do not fit on one Device
- benefit from pipeline partitioning
- benefit from Operator-level placement
- use separate Devices for prefill and decode
- replicate selected weights for locality
- maintain Device-local KV state
- use asymmetric Device capabilities
- need explicit memory-capacity partitioning

The multi-Device contract must not accidentally introduce distributed-system
semantics into Magnetar Core.

This change therefore scopes itself to local multi-Device execution.

It does not define:

- multi-host execution
- distributed consensus
- distributed scheduling
- NCCL
- MPI
- RDMA
- all-reduce
- all-gather
- reduce-scatter
- tensor-parallel collective protocols

Those require a separate future contract.

The architectural invariant remains:

```text
Runtime decides placement.

Memory Manager decides resource residency and partition policy.

Provider realizes execution and movement.

Device describes hardware.

Scheduler admits and orders work.

Tachyon, if present, owns distributed orchestration outside Magnetar.
```

## What Changes

This change defines:

- MultiDevicePlacementPlan
- PlacementDomain
- PlacementBinding
- DeviceSet
- DeviceCompatibilitySet
- Operator placement
- execution-segment placement
- model-weight partitioning
- weight replication
- Tensor partition descriptors
- TensorPartitionAxis
- TensorShard
- partition identity
- partition completeness
- partition reconstruction metadata
- local pipeline partitioning
- stage boundaries
- explicit inter-Device movement
- peer access and peer transfer integration
- host-staging policy preservation
- per-Device memory-pool budgets
- allocation-plan partitioning
- KV-cache ownership
- Device-local KV placement
- prefill/decode placement
- placement guards
- placement readiness
- re-placement
- Device failure handling
- degraded execution
- fallback placement
- Plan generation replacement
- observability
- conformance

## Scope

The baseline multi-Device model is:

```text
one Runtime
one host
one Model Instance
multiple local Devices
```

Example:

```text
Runtime
 |
 +-- Provider CUDA
 |      |
 |      +-- Device GPU0
 |      +-- Device GPU1
 |
 +-- Provider CPU
        |
        +-- Device CPU0
```

A single Prepared Execution Plan MAY use several of these Devices.

## Multi Device Is Runtime Placement

Model Components SHALL continue describing portable computation.

They SHALL NOT choose concrete Devices.

For example, a Model Component may expose:

```text
Transformer Block 0
Transformer Block 1
Transformer Block 2
...
```

but SHALL NOT declare:

```text
Block 0 -> GPU0
Block 1 -> GPU1
```

as authoritative hardware placement.

Concrete placement belongs to Runtime policy.

## DeviceSet

Runtime SHALL represent a set of Devices available for one placement decision.

Conceptually:

```text
DeviceSet
    id
    devices
    Provider bindings
    compatibility metadata
```

A DeviceSet SHALL not imply all Devices have identical capability.

## PlacementDomain

A PlacementDomain defines a logical set of compatible execution resources
available to one placement scope.

A PlacementDomain MAY include:

- one Device
- several Devices owned by one Provider
- several Devices owned by different Providers where explicit movement is
  permitted

The baseline SHOULD favor same-Provider multi-Device placement where possible.

## Placement Binding

A PlacementBinding associates one logical execution/resource scope with a
concrete Provider/Device target.

Conceptually:

```text
PlacementBinding
    ProviderBinding
    DeviceBinding
    memory_domain
    constraints
```

It SHALL not expose native Device handles.

## MultiDevicePlacementPlan

Runtime SHALL represent local multi-Device placement as an explicit plan.

Conceptually:

```text
MultiDevicePlacementPlan
    id
    generation
    DeviceSet
    graph_bindings
    resource_bindings
    movement_edges
    memory_budgets
    guards
    policy_fingerprint
```

## Placement Plan Identity

Placement Plan identity SHOULD incorporate materially relevant placement
inputs, including:

- Execution Graph fingerprint
- Model Instance revision
- Device stable identities
- Provider versions
- Device capabilities
- memory capacity classes
- placement policy version
- partition metadata
- relevant Kernel requirements

It SHALL NOT contain native pointers.

## Placement Granularity

Runtime MAY place work at several granularities:

- Model Instance
- graph segment
- Transformer layer
- Operator group
- individual Operator

The baseline SHOULD prefer stable coarse placement over excessively fine
cross-Device movement.

## Placement Cost

Placement policy SHOULD account for total execution cost rather than Kernel
latency alone.

Relevant factors MAY include:

- Kernel performance
- Device memory capacity
- current Device pressure
- peer-transfer cost
- host-staging cost
- residency
- workspace demand
- KV locality
- placement stability
- synchronization overhead

## No Hidden Cross Device Movement

A placement that requires Resource movement between Devices SHALL make that
movement explicit.

For example:

```text
segment GPU0
     |
Tensor
     |
explicit movement
     |
segment GPU1
```

Runtime SHALL NOT treat two Devices as one transparent memory space unless an
explicit capability provides that semantic.

## Placement Strategies

Runtime MAY support placement strategies such as:

```text
single-device
pipeline
layer-range
operator-group
capacity-balanced
latency-aware
memory-aware
pinned
```

The strategy vocabulary SHALL remain extensible.

These are Runtime policy strategies, not Provider identities.

## Single Device Is Valid Multi Device Baseline

A placement system SHALL remain capable of selecting one Device only.

Supporting multi-Device placement SHALL NOT require partitioning every model.

## Pipeline Partitioning

The baseline multi-Device execution strategy SHOULD support local pipeline
partitioning.

Example:

```text
GPU0
  blocks 0..15
       |
       v
GPU1
  blocks 16..31
```

The inter-stage activation movement SHALL be explicit.

## Pipeline Stage

Runtime MAY represent a PipelineStage.

Conceptually:

```text
PipelineStage
    stage_id
    graph_nodes
    PlacementBinding
    input_requirements
    output_requirements
```

PipelineStage is a Runtime execution-planning concept.

It SHALL not become Model Component hardware authority.

## Pipeline Stage Ordering

Pipeline stages SHALL preserve Execution Graph semantics.

A downstream stage SHALL depend on completion/readiness of the upstream stage
and any required data movement.

## Pipeline Overlap

Runtime MAY overlap work from different requests/stages when:

- resource dependencies permit
- Scheduler policy permits
- Device memory capacity permits
- ExecutionStream dependencies are satisfied

This change does not mandate pipeline micro-batching.

## Weight Placement

Model weights MAY be:

- resident entirely on one Device
- partitioned across Devices
- replicated across Devices
- partly replicated and partly partitioned

Runtime/Memory Manager owns this decision.

## Weight Partition

A weight partition SHALL preserve relationship to the logical Model Artifact
tensor.

Conceptually:

```text
LogicalWeight
    |
    +-- partition 0 -> GPU0
    +-- partition 1 -> GPU1
```

Partition metadata SHALL identify which logical region each partition
represents.

## Weight Replication

Immutable weights MAY be replicated when memory capacity permits.

A replica SHALL retain:

- logical source identity
- artifact digest relationship
- dtype
- layout
- revision
- validity

Replication SHALL not create a different semantic Model Artifact.

## Tensor Partition Descriptor

Magnetar SHALL support a TensorPartitionDescriptor describing how a logical
Tensor Resource is partitioned.

Conceptually:

```text
TensorPartitionDescriptor
    logical_resource
    partition_axis
    partition_count
    shards
    reconstruction
```

## Tensor Partition Axis

The partition descriptor SHOULD support explicit logical axes.

Examples:

```text
dimension 0
dimension 1
head axis
hidden axis
vocabulary axis
```

The representation SHALL remain based on logical tensor semantics.

It SHALL NOT encode vendor-native Device partitioning.

## Tensor Shard

A TensorShard represents one logical partition of a Tensor Resource.

Conceptually:

```text
TensorShard
    shard_id
    parent_resource
    logical_range
    shape
    dtype
    layout
    PlacementBinding
    residency
```

## Shard Is A Resource Relationship

A TensorShard MAY be represented as:

- a View over existing Resource storage
- a separately allocated Resource
- an immutable Model weight partition

The descriptor SHALL make the relationship explicit.

## Shard Bounds

Shard ranges SHALL be validated.

Overlaps or gaps SHALL be explicit according to partition semantics.

## Complete Partition

A complete non-replicated partition SHALL cover the intended logical tensor
domain exactly according to its descriptor.

Unexpected gaps or overlaps SHALL be rejected.

## Replicated Partition

A replicated tensor SHALL explicitly identify equivalent copies.

Replica semantics SHALL not be confused with partition semantics.

## Hybrid Partition

Runtime MAY support a logical tensor where some regions are partitioned and
some are replicated.

Such structure SHALL be explicit rather than inferred.

## Partition Reconstruction

Partition metadata SHOULD contain enough logical information to reconstruct or
reason about the whole Tensor where required.

This does not mean Runtime must physically reconstruct the Tensor on host.

## Partition Does Not Imply Collective

A partitioned Tensor SHALL NOT automatically imply an all-gather, all-reduce,
or other collective.

Required communication SHALL be explicitly represented by the execution
strategy.

## Operator Partition Compatibility

A Kernel SHALL only consume Tensor shards directly if its Operator/Kernel
contract supports the partitioned input semantics.

Otherwise Runtime SHALL:

- choose another Kernel
- materialize/reconstruct explicitly
- perform another placement
- fail

It SHALL NOT silently reinterpret a shard as the complete Tensor.

## Partition-Aware Kernel

Kernel metadata MAY declare support for specific partitioned input/output
semantics.

This metadata SHALL remain explicit.

It SHALL not introduce distributed collective semantics unless a future
collective contract defines them.

## Partition-Aware Execution Graph

Execution Graph MAY contain portable explicit partition/reassembly operations
only where their semantics are defined.

Runtime placement itself SHALL not rewrite graph semantics silently.

## Cross Device Resource Movement

Movement between local Devices SHALL use existing explicit data-movement and
ExecutionStream contracts.

Possible implementations include:

- direct peer copy
- direct peer access
- Provider-managed local transfer
- host-staged transfer if explicitly permitted

## Peer Access

Peer access MAY avoid a copy if Provider/Device capability and policy permit it.

Peer-access capability SHALL be validated per Device pair.

## Peer Access Is Not Placement

The fact that GPU1 can read GPU0 memory SHALL not mean every GPU1 Kernel should
do so.

Placement policy MAY prefer local replicas/transfers for performance.

## Peer Transfer

A direct peer copy remains an explicit movement operation.

Zero host staging does not mean zero movement.

## Host Staging

If inter-Device movement requires host staging, host-staging policy remains
authoritative.

A forbidden host-staging route SHALL not be used silently.

## Cross Provider Placement

One Placement Plan MAY use multiple Providers only where Resource movement and
execution contracts permit it.

Example:

```text
CUDA GPU
   |
explicit host-visible boundary
   |
CPU Provider
```

Baseline policy SHOULD be conservative because cross-Provider synchronization
is Runtime-mediated.

## Provider Native Interop

Native peer/interoperability handles SHALL not be exchanged through generic
Core placement contracts.

Future explicit interoperability capabilities may optimize this.

## Device Local Memory Budget

Each participating Device SHALL have explicit or derived memory budget.

Memory budget SHOULD include:

- weights
- KV
- workspace
- transient memory
- transfer buffers
- reserved headroom

## Per Device Pools

Memory Manager MAY create independent logical memory pools per Device.

Example:

```text
GPU0
  weights pool
  KV pool
  workspace pool

GPU1
  weights pool
  KV pool
  workspace pool
```

## Pool Partitioning

A multi-Device AllocationPlan SHALL identify which Device pool satisfies each
allocation slot.

It SHALL not allocate from an unspecified global Device memory pool.

## Memory Balance

Placement policy SHOULD consider memory capacity and pressure independently for
each Device.

Equal layer counts SHALL not be assumed optimal if Devices have different
capacity/performance.

## Heterogeneous Devices

A DeviceSet MAY contain heterogeneous Devices.

Placement MAY account for:

- architecture
- memory capacity
- supported dtypes
- supported Kernels
- performance
- peer topology

Runtime SHALL not assume homogeneity.

## Device Compatibility

A graph segment may only be placed on a Device for which required:

- Kernels
- memory domains
- dtype/layout
- execution capabilities
- workspace
- synchronization

are compatible.

## Placement Eligibility

Placement SHALL apply hard eligibility before optimization.

Conceptually:

```text
eligible(Device, segment) =
    Provider supports required Kernels
 && Device supports required features
 && memory feasible
 && Resource Affinity valid
 && required movements permitted
 && policy allows Device
```

Only eligible placements may be ranked.

## Placement Ranking

Among eligible placements Runtime MAY optimize:

- latency
- throughput
- memory balance
- transfer cost
- energy
- pressure
- stability

## Transfer-Aware Placement Cost

Placement SHALL be allowed to account for movement cost.

Example:

```text
Kernel on GPU1 = 20 us
required transfer = 80 us

Kernel on GPU0 = 40 us
no transfer
```

GPU0 may be the better placement.

## Placement Stability

Runtime SHOULD support hysteresis to avoid moving graph segments between
Devices on small pressure/performance fluctuations.

## Placement Pinning

Deployment policy MAY pin:

- Model Instance to DeviceSet
- stage to Device
- weights to Device
- KV to Device

A pin SHALL not bypass compatibility or Device failure.

## Prefill Decode Placement

Prefill and decode MAY use different Device placement plans.

For example:

```text
prefill -> GPU0 + GPU1 pipeline
decode  -> GPU1
```

This MAY require explicit state/Resource movement between phases.

## Phase Transition

A prefill-to-decode placement transition SHALL ensure:

- KV state is available on decode Device(s)
- required weights are available
- pending prefill work completed
- explicit movements completed
- decode Prepared Plan guards pass

## KV Cache Ownership

Each KV page or partition SHALL have explicit Device residency/ownership.

Baseline local decode SHOULD favor keeping one sequence's required KV near the
Device executing its Attention kernels.

## KV Device Locality

Placement SHOULD avoid unnecessary per-token KV movement between Devices.

A sequence SHOULD NOT bounce between Devices every token without explicit
policy and demonstrated benefit.

## KV Partitioning

KV cache MAY be partitioned across Devices only when the selected Attention
execution contract supports that partitioning.

This change does not define multi-Device Attention collectives.

## KV Replication

KV replication MAY be supported for specific policies, but replication
coherency/update semantics SHALL be explicit.

Baseline SHOULD prefer one authoritative Device-local KV ownership where
possible.

## Session Placement Affinity

A Session MAY acquire a preferred Device/placement affinity.

The preference MAY improve KV locality and plan reuse.

It SHALL not become an immutable hardware requirement if failure policy allows
migration.

## Session Migration

Moving a Session to another Device SHALL be an explicit placement transition.

It MAY require moving:

- KV pages
- adapter resources
- session-specific buffers

All movement SHALL preserve ResourceReadiness.

## Model Instance Placement

A Model Instance MAY own one or more MultiDevicePlacementPlans.

Different plans MAY serve different:

- workload buckets
- batch sizes
- phases
- degraded/fallback modes

## Prepared Execution Plan Integration

PreparedExecutionPlan SHALL capture concrete Device bindings derived from a
MultiDevicePlacementPlan.

Example:

```text
PreparedExecutionPlan
 |
 +-- stage 0 -> GPU0
 +-- transfer
 +-- stage 1 -> GPU1
```

## Exact Placement Binding

A ready Plan SHALL not silently migrate one segment to another Device without a
new Plan generation or explicit dynamic-placement contract.

Baseline behavior SHOULD use a new Plan generation.

## Placement Guards

Prepared Plan SHALL validate placement-critical assumptions.

Guards MAY include:

- Device available
- Provider available
- required Kernel prepared
- required Resource residency
- memory reservation valid
- peer path available where required
- host-staging policy unchanged

## Placement Staleness

A Plan MAY become stale but still safe when:

- another Device becomes more attractive
- pressure changes
- performance feedback changes ranking
- a better weight replica becomes available

Staleness MAY trigger background re-placement.

## Placement Hard Invalidation

A Plan SHALL be invalid for new work when:

- Device is lost
- Provider is unavailable
- required peer path disappears
- memory reservation becomes impossible
- Resource Affinity hard constraint fails
- required Kernel becomes unavailable/revoked

## Re Placement

Runtime MAY build a replacement MultiDevicePlacementPlan.

Re-placement SHALL occur outside active Kernel hot path.

## No Hot Path Global Re Placement

A token decode SHALL NOT synchronously recompute arbitrary multi-Device
placement because Device pressure changed slightly.

Runtime MAY:

- continue current valid Plan
- use ready fallback Plan
- request background re-placement
- fail if current Plan is hard-invalid

## Atomic Placement Replacement

A replacement Prepared Execution Plan SHALL be published atomically.

In-flight work SHALL retain the old Plan generation until safe completion.

## Device Failure Domain

Runtime SHALL treat each Device as a potential local failure domain.

Loss of one Device SHALL not automatically imply all Providers/Devices are
lost.

However, a Plan depending on that Device becomes invalid.

## Degraded Placement

Runtime MAY support a degraded Plan using fewer Devices.

Example:

```text
normal:
    GPU0 + GPU1

degraded:
    GPU0 only
```

The degraded Plan SHALL independently satisfy:

- model capacity
- Kernel availability
- memory
- correctness
- policy

## No Implicit Degraded Mode

Runtime SHALL not assume remaining Devices can run the model after failure.

A valid fallback/degraded Plan must exist or be successfully built.

## Device Recovery

A recovered Device SHALL not automatically rejoin active placement.

It SHALL undergo normal:

- health/readiness
- Provider compatibility
- memory pool setup
- Kernel preparation
- Plan construction

before new work uses it.

## Scheduler Boundary

Scheduler MAY be aware of logical placement constraints such as:

- Session affinity
- DeviceSet availability
- per-Device pressure
- Plan readiness

Scheduler SHALL not own native Device handles or peer-transfer mechanisms.

## Admission

Admission SHALL consider aggregate and per-Device resources.

A Model Instance requiring two Devices SHALL not be considered ready if one
mandatory Device lacks required capacity.

## Multi Device Concurrency

Independent segments or requests MAY execute concurrently on different Devices.

ExecutionStream and CompletionToken semantics SHALL preserve dependencies.

## Cross Device Completion

An inter-Device movement or downstream stage SHALL depend on upstream
CompletionToken and movement completion as required.

## Failure Propagation

Failure of an upstream Device/segment SHALL prevent dependent downstream
execution unless explicit recovery/fallback exists.

## Resource Lifetime

Resources participating in inter-Device transfer SHALL remain alive until
source/destination operations complete.

## Replication Lifecycle

A Resource replica MAY be evicted independently if:

- another authoritative/valid copy remains
- no in-flight work references replica
- placement plans depending on it are updated/invalidated appropriately

## Weight Replica Failure

Loss of one immutable weight replica does not invalidate the logical Model
Artifact if another compatible replica/source exists.

It MAY invalidate Plans bound to the lost replica.

## Placement And Kernel Selection

Kernel selection and Device placement are interdependent.

Runtime MAY solve them iteratively or jointly.

The architecture does not mandate one optimization algorithm.

The hard invariant is that final Plan contains compatible:

```text
Operator
Kernel
Provider
Device
Resource residency
Memory plan
```

bindings.

## Placement And Autotuning

Autotuning evidence SHALL be target-specific.

A specialization tuned on GPU0 SHALL not automatically be treated as optimal on
GPU1 if compatibility/performance context differs.

## Placement And Performance Feedback

Online Performance Model MAY provide per-Device placement evidence.

A Device/segment performance regression MAY request re-placement.

It SHALL not generate arbitrary code.

## Placement Cache

Runtime MAY cache MultiDevicePlacementPlans.

The cache SHALL be distinct from:

- Kernel Artifact Cache
- Autotuning Cache
- Allocation Plan Cache
- Prepared Execution Plan Cache

## Placement Cache Key

A key SHOULD include:

- graph fingerprint
- Model Instance revision
- DeviceSet fingerprint
- Provider versions
- Device capability classes
- memory budget classes
- workload scope
- placement policy version
- required partition descriptors

## Cached Placement Revalidation

A cached Placement Plan SHALL be revalidated against:

- current Device availability
- Provider readiness
- memory capacity
- peer capability
- required Kernel availability
- current policy
- Resource residency

## Native Handle Privacy

Placement contracts SHALL NOT expose:

- CUDA device pointers
- native queue handles
- peer memory handles
- PCI BAR mappings
- OS handles
- Provider-native Device handles

## WIT Boundary

WASM Components SHALL not select concrete Devices or multi-Device topology.

Components MAY expose portable graph/resource requirements only.

## Runtime Inference API

Normal inference callers MAY express high-level preferences such as:

- low latency
- deterministic
- memory constrained

They SHALL NOT directly force:

```text
layer 0 -> GPU0
layer 1 -> GPU1
```

through ordinary generation requests.

Administrative/deployment policy MAY define explicit placement outside normal
inference request authority.

## Error Model

Structured errors SHOULD include:

```text
multi-device-placement-disabled
multi-device-placement-policy-invalid
multi-device-placement-no-devices
multi-device-placement-no-feasible-plan
multi-device-placement-device-incompatible
multi-device-placement-provider-incompatible
multi-device-placement-memory-infeasible
multi-device-placement-kernel-unavailable
multi-device-placement-affinity-invalid

multi-device-placement-transfer-denied
multi-device-placement-peer-access-unavailable
multi-device-placement-peer-transfer-unavailable
multi-device-placement-host-staging-denied

multi-device-placement-plan-stale
multi-device-placement-plan-invalidated
multi-device-placement-plan-build-failed
multi-device-placement-plan-cache-stale

tensor-partition-invalid
tensor-partition-axis-invalid
tensor-partition-gap
tensor-partition-overlap
tensor-partition-shard-missing
tensor-partition-shard-incompatible
tensor-partition-kernel-unsupported

multi-device-kv-placement-invalid
multi-device-kv-migration-failed

multi-device-stage-invalid
multi-device-stage-dependency-invalid
multi-device-stage-transfer-failed

multi-device-device-lost
multi-device-degraded-plan-unavailable
multi-device-replacement-failed

internal-multi-device-placement-error
```

## Observability

Multi-Device observability MAY include:

```text
placement-plan-build-started
placement-candidate-evaluated
placement-candidate-excluded
placement-plan-ready

tensor-partition-created
weight-replica-created
weight-replica-evicted

stage-submitted
stage-completed
cross-device-transfer-started
cross-device-transfer-completed

session-placement-bound
session-placement-migrated

placement-plan-stale
placement-plan-invalidated
placement-rebuild-requested
degraded-placement-activated
device-lost
device-recovered
```

Observability MAY report:

- logical Placement Plan ID/generation
- stable Provider/Device identities
- graph-stage IDs
- Tensor partition IDs
- shard counts
- movement byte counts
- peer/direct/staged movement class
- per-Device memory summaries
- placement decision reasons
- fallback/degraded state

Observability SHALL NOT expose:

- Device pointers
- native peer handles
- native streams
- native synchronization objects
- model-weight contents
- KV contents
- prompts
- secrets
- credentials

## Conformance

Conformance SHALL validate:

- Model Component cannot choose concrete Devices
- Runtime owns placement
- Device remains descriptive
- multi-Device Plan makes Device bindings explicit
- Tensor partitions validate bounds/completeness
- replicas are distinct from partitions
- shard is not silently treated as complete Tensor
- unsupported partition-aware Kernel is rejected
- cross-Device movement is explicit
- host staging prohibition is preserved
- peer access requires explicit capability
- peer access does not automatically imply optimal placement
- per-Device memory budgets are enforced
- heterogeneous Device capability is respected
- transfer cost can influence placement
- Prepared Plan captures exact placement
- Plan does not silently migrate mid-flight
- KV locality is preserved
- Session migration moves required state explicitly
- Device loss invalidates dependent Plan
- degraded mode requires valid fallback Plan
- recovered Device undergoes normal readiness
- cached Placement Plan is revalidated
- native handles remain private
- observability is redacted

## Non-Goals

This change does not:

- define multi-host execution
- define distributed consensus
- define distributed Scheduler
- define Tachyon-specific placement RPC
- define NCCL
- define MPI
- define RDMA
- define all-reduce
- define all-gather
- define reduce-scatter
- define tensor-parallel collective semantics
- define expert parallelism
- require pipeline parallelism
- require homogeneous GPUs
- let Components select Devices
- move placement authority into Provider
- make Device an orchestration object

## Impact

Magnetar gains a local multi-Device execution model:

```text
                    Runtime
                       |
             Placement Planner
                       |
           +-----------+-----------+
           |                       |
           v                       v
        GPU0                     GPU1
    weights/blocks           weights/blocks
    KV / workspace           KV / workspace
           |                       ^
           |                       |
           +--- explicit move -----+
```

The model can therefore scale beyond one Device while preserving the existing
Provider, Memory Manager, Resource Affinity, Prepared Plan, and synchronization
boundaries.