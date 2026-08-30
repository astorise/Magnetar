# device-resident-resource Specification

## Purpose
This specification defines device-resident-resource requirements.
## Requirements
### Requirement: Resource Residency Is Explicit

Runtime SHALL be able to determine the logical residency of a Tensor Resource.

#### Scenario: GPU weight Tensor

Given model weight is allocated on GPU Device memory

When Runtime inspects its placement

Then Resource residency identifies compatible Provider/Device memory domain
without exposing native address.

### Requirement: Device Residency Does Not Require Host Representation

Tensor Resource SHALL be able to exist and execute entirely Device-side without an
authoritative host byte buffer.

#### Scenario: Intermediate activation

Given MatMul produces Tensor consumed only by Device Kernels

When inference proceeds

Then Runtime does not require host-side materialization of the activation.

### Requirement: Same-Device Pipeline Avoids Mandatory Host Copy

Compatible consecutive Device Kernels SHALL be able to consume the same
Device-resident Resource.

#### Scenario: MatMul to RMSNorm

Given both Kernels execute on GPU0

And layout/dtype are compatible

When RMSNorm consumes MatMul output

Then no host round-trip is required.

### Requirement: Zero Copy Is Explicitly Eligible

A consumer SHALL access Resource zero-copy only when compatibility and policy
permit it.

#### Scenario: Incompatible layout

Given Resource is Device-local but Kernel requires another layout

When zero-copy eligibility is checked

Then Device residency alone is insufficient.

### Requirement: Explicit Transfer For Residency Change

Changing authoritative storage between incompatible memory domains SHALL use
explicit data movement.

#### Scenario: GPU output required by CPU Kernel

Given no shared-access capability exists

When CPU Kernel needs Tensor

Then Runtime performs authorized explicit transfer rather than implicit access.

### Requirement: Host Staging Policy Is Preserved

Residency optimization SHALL not bypass host-staging prohibition.

#### Scenario: GPU0 to GPU1 requires host staging

Given movement policy forbids host staging

When no peer path exists

Then transfer is denied or alternate execution path is selected.

### Requirement: Residency Survives Asynchronous Execution

Resource SHALL remain resident/valid while in-flight work requires its storage.

#### Scenario: Resource eviction requested

Given pending CompletionToken references Resource

When pressure requests eviction

Then physical storage is retained until safe.

### Requirement: Zero Copy Does Not Expose Native Address

Zero-copy access SHALL not require exposing native Device pointer through
Runtime public contracts.

#### Scenario: CUDA Resource

Given Kernel can directly consume Device allocation

When Runtime binds it

Then Provider resolves native pointer privately.

### Requirement: Resource Replica Validity Is Per Device

Replicated Resource copies SHALL track validity independently.

#### Scenario: GPU1 weight replica evicted

Given GPU0 replica remains valid

When GPU1 copy is removed

Then logical weight remains valid but Plans bound to GPU1 replica may stale or
invalidate.

### Requirement: Cross Device Access Requires Capability Or Movement

A Device SHALL not directly consume another Device's resident Resource unless
peer/direct-access capability permits it.

#### Scenario: GPU1 accesses GPU0 Resource

Given peer-read unavailable

When direct binding is attempted

Then Runtime requires explicit transfer/replica.

### Requirement: Peer Zero Copy Preserves Affinity And Readiness

Direct peer access SHALL still satisfy Resource Affinity and ResourceReadiness.

#### Scenario: GPU0 write pending

Given GPU1 can peer-read GPU0 memory

When GPU1 consumer is submitted

Then dependency on GPU0 write completion remains required.

### Requirement: Device Replica Creation Is Explicit

Creating Resource replica on another Device SHALL be an explicit residency
operation.

#### Scenario: Weight pre-replication

Given model load chooses copies on GPU0/GPU1

When replicas are created

Then transfer/allocation is represented and auditable.

### Requirement: Replica Eviction Is Completion Safe

A Device replica SHALL not be evicted while in-flight work references it.

#### Scenario: Plan retirement

Given GPU1 stage still executes

When pressure targets weight replica

Then eviction waits for safe completion.

