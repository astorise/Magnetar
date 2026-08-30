## ADDED Requirements
### Requirement: Memory Manager Owns Pool Policy

Memory Manager SHALL decide logical pool classes, capacity, reservations,
borrowing, pressure, reclaim, and allocation policy.

#### Scenario: Provider has native allocator

Given Provider offers native memory-pool API

When Runtime configures memory policy

Then Provider API does not become the policy authority.

### Requirement: Allocation Request Is Logical

Memory allocation SHALL begin from logical requirements rather than native
allocator arguments.

#### Scenario: Attention workspace

Given Prepared Kernel requires 32 MiB aligned to 256 bytes

When Runtime requests storage

Then Memory Manager expresses bytes/alignment/domain/class rather than native
CUDA allocation flags.

### Requirement: Allocation Lease Has No Pointer Semantics

AllocationLease SHALL refer to pool-backed storage without exposing Device
pointer.

#### Scenario: Lease created

Given Provider realizes backing

When Runtime stores lease

Then identity remains logical and opaque.

### Requirement: Temporal Reuse Is Lifetime Safe

Storage SHALL be reused by distinct Tensor Resources whose lifetimes do not
overlap.

#### Scenario: Intermediate A ends before C begins

Given asynchronous completion confirms A is no longer used

When C is allocated

Then C SHALL reuse A's backing region.

### Requirement: Asynchronous Completion Governs Reuse

Graph topology alone SHALL NOT authorize storage reuse where work remains
in-flight.

#### Scenario: Node A logically passed

Given its Kernel is still executing

When planner considers storage reuse

Then AllocationLease remains unavailable until completion.

### Requirement: Fragmentation Is Distinct From Capacity Exhaustion

Runtime SHALL distinguish fragmented free memory from true total-capacity
exhaustion.

#### Scenario: Large contiguous requirement

Given 512 MiB total free exists in small regions

But 256 MiB compatible contiguous block cannot be realized

When allocation fails

Then failure SHALL be classified as fragmentation.

### Requirement: Compaction Is Policy Controlled

Memory relocation/compaction SHALL occur only under Memory Manager control.

#### Scenario: Fragmented transient pool

Given enough movable idle Resources exist

When policy allows compaction

Then Runtime SHALL relocate them safely.

### Requirement: In-Flight Resource Is Not Movable

Resource referenced by unfinished Device work SHALL not be relocated
incompatibly.

#### Scenario: Kernel pending

Given Tensor lease is active in Device execution

When compaction runs

Then Resource is skipped/pinned until completion.

### Requirement: Active Mapping Pins Backing

Resource with active host mapping SHALL not be relocated or reused
incompatibly.

#### Scenario: Host reads mapped logits

Given mapping remains open

When compaction occurs

Then backing remains valid.

### Requirement: Fixed Address Requirement Is Honored

A Provider-prepared artifact requiring stable address SHALL cause relevant
allocation to remain pinned for that lifetime.

#### Scenario: Captured graph binds fixed buffer

Given Provider declares address stability requirement

When Memory Manager plans buffers

Then backing is not relocated while capture is active.