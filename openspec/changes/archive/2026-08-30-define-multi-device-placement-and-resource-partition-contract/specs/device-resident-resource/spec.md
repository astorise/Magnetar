## ADDED Requirements
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