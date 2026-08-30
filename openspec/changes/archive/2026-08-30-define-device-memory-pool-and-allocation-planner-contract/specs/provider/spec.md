## ADDED Requirements
### Requirement: Provider Realizes Memory Pool Backing

Provider SHALL allocate native blocks or use native memory-pool facilities to
realize Runtime logical DeviceMemoryPools.

#### Scenario: CUDA Provider

Given Runtime requests Device-local pool backing

When Provider realizes it

Then CUDA-specific pool/allocation state remains Provider-private.

### Requirement: Provider Advertises Allocation Capabilities

Provider SHALL advertise allocation characteristics relevant to Runtime
planning.

#### Scenario: Alignment capability

Given Device allocation requires minimum granularity

When capabilities are queried

Then Runtime can plan compatible slots.

### Requirement: Provider Does Not Decide Pool Reservations

Provider SHALL NOT override Runtime hard/soft pool reservations.

#### Scenario: Provider has spare memory

Given Runtime protects KV reservation

When Provider receives optional workspace request

Then it cannot silently consume reserved capacity outside Runtime decision.

### Requirement: Provider Supports Opaque Block Realization

Provider-backed AllocationBlock identity SHALL remain opaque to Runtime.

#### Scenario: Native block created

Given Provider allocates large Device region

When Runtime receives block token

Then token has no pointer semantics.

### Requirement: Provider Reports Native Allocation Failure Structurally

Native allocation failure SHALL be normalized into structured Provider/Memory
error.

#### Scenario: Driver allocation fails

Given pool growth requests new block

When driver rejects allocation

Then Runtime receives provider-allocation-failed or equivalent structured state.

### Requirement: Provider Advertises Address Stability Constraints

Provider SHALL report when prepared Kernel/segment requires stable backing
addresses.

#### Scenario: Native graph capture

Given graph cannot tolerate relocated buffers

When Plan is prepared

Then Memory Manager receives explicit non-movable requirement.