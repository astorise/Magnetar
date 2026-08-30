## ADDED Requirements
### Requirement: Memory Manager Owns Residency Policy

Memory Manager SHALL remain authoritative for logical Resource placement,
residency, replication, eviction, and movement authorization.

#### Scenario: Provider prefers host-visible allocation

Given Runtime policy requires Device-local latency-critical weights

When allocation is planned

Then Provider preference does not override Memory Manager policy.

### Requirement: Provider Realizes Native Allocation

Memory Manager SHALL request logical memory capability while Provider realizes the
native allocation.

#### Scenario: Device-local allocation

Given Runtime needs GPU-local storage

When allocation occurs

Then Provider may use native Device allocator privately.

### Requirement: Memory Manager Tracks Replicas

Multiple physical copies SHALL have explicit validity state.

#### Scenario: GPU0 and GPU1 weights

Given immutable weight is replicated

When either Device reads

Then Runtime knows both copies are current.

### Requirement: Memory Manager Tracks Mapping Lifetime

Mapped Resource SHALL remain allocated and compatible for mapping duration.

#### Scenario: Host read mapping

Given Device Resource is mapped

When eviction occurs concurrently

Then eviction is delayed or mapping fails safely.

### Requirement: Eviction Is Completion-Aware

Resource SHALL not be evicted while in-flight Device access exists.

#### Scenario: Pending Attention

Given KV page is active

When Device pressure triggers eviction

Then eviction waits for completion.

### Requirement: Spill Is Explicit

Moving Resource to lower-priority memory domain SHALL be represented as explicit
residency transition/data movement.

#### Scenario: Device pressure

Given inactive cache entry spills to host

When spill occurs

Then host staging policy and transfer completion are enforced.

### Requirement: Residency Pinning Is Bounded

Memory Manager SHALL support pinning Resources according to policy.

#### Scenario: Active KV cache

Given low-latency session pins KV to GPU

When capacity is exhausted

Then new admission may fail rather than silently violating pin.

### Requirement: Allocation Reuse Preserves Aliasing And Mapping

Underlying storage SHALL not be recycled while Views, mappings, or in-flight
operations require it.

#### Scenario: Parent Tensor dropped

Given live View still exists

When Memory Manager evaluates storage

Then allocation remains alive.
