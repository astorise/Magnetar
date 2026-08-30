## ADDED Requirements
### Requirement: Multi Device Memory Is Accounted Per Device

Memory Manager SHALL track capacity and reservations independently for each
participating Device.

#### Scenario: GPU1 low capacity

Given GPU0 has free memory but GPU1 pool is exhausted

When GPU1 stage requests workspace

Then GPU0 free capacity does not automatically satisfy GPU1 request.

### Requirement: Placement Planner Consumes Memory Feasibility

Multi-Device placement SHALL incorporate Device-specific memory feasibility.

#### Scenario: Layer range too large

Given candidate stage exceeds GPU1 pool

When placement candidates are evaluated

Then candidate is excluded.

### Requirement: Allocation Plan Binds Slots To Device

Every Device-local allocation slot SHALL identify target Device/pool.

#### Scenario: Attention workspace

Given stage executes GPU1

When AllocationPlan is inspected

Then its workspace slot is bound to GPU1-compatible pool.

### Requirement: Weight Replication Consumes Explicit Capacity

Replicating weights SHALL consume and reserve memory independently on each
Device.

#### Scenario: 4 GiB weight replicated twice

Given GPU0 and GPU1 each receive replica

When capacity is accounted

Then each Device accounts 4 GiB.

### Requirement: Weight Partition Accounts Actual Shards

Partitioned weights SHALL account only the storage hosted by each Device plus
allocator overhead.

#### Scenario: Half split

Given 8 GiB weight split equally

When placement accounting runs

Then approximately 4 GiB payload is assigned per Device.

### Requirement: Device Failure Does Not Free In Flight Memory Unsafely

Failure handling SHALL preserve conservative lifetime semantics.

#### Scenario: GPU1 lost

Given completion state is uncertain

When pool teardown occurs

Then Runtime does not assume resources are safely reusable on same native
context.

### Requirement: Re Placement Revalidates Memory Reservations

Replacement placement SHALL acquire required capacity before becoming READY
where policy requires reservation.

#### Scenario: Move stage to GPU0

Given GPU0 lacks memory for transferred weights/workspace

When fallback Plan builds

Then it does not become ready.