## ADDED Requirements
### Requirement: Logical Device Memory Pool

Runtime Memory Manager SHALL support logical DeviceMemoryPools for reusable
Device-backed storage.

#### Scenario: GPU inference pool

Given GPU Device has available compatible memory

When Runtime initializes inference capacity

Then it SHALL create logical weights, KV, workspace, and transient pools without
exposing native allocator objects.

### Requirement: Pool Does Not Expose Native Address

A DeviceMemoryPool SHALL NOT expose backing native memory address as pool
identity.

#### Scenario: Provider uses large CUDA allocation

Given pool internally maps to Device allocation

When Runtime inspects pool

Then CUDA pointer is absent.

### Requirement: Pool Capacity Is Accounted

Runtime SHALL track logical pool capacity and usage.

#### Scenario: KV pool pressure

Given KV pages consume 7 GiB of 8 GiB configured pool

When capacity is queried

Then committed/leased/reclaimable state is available.

### Requirement: Hard Reservation Is Protected

Hard-reserved capacity SHALL not be silently consumed by unrelated lower
priority allocation classes.

#### Scenario: Decode KV reservation

Given 2 GiB remain reserved for KV

When optional autotuning requests 2 GiB workspace

Then Memory Manager SHALL deny it rather than consume protected KV capacity.

### Requirement: Soft Reservation SHALL Be Borrowed Explicitly

Soft-reserved capacity SHALL be borrowed according to policy.

#### Scenario: Idle workspace pool

Given workspace reservation is unused

When KV temporarily needs capacity

Then borrowing SHALL occur if policy permits and accounting records it.

### Requirement: Watermarks Drive Pressure State

Pool pressure SHALL be derivable from configured watermarks.

#### Scenario: High watermark crossed

Given usage exceeds configured high watermark

When Memory Manager updates pool state

Then reclamation/admission policy SHALL activate.

### Requirement: Pending Reclaim Is Not Free

Memory waiting for asynchronous completion SHALL not be counted as immediately
available.

#### Scenario: Workspace logically released

Given Kernel still runs

When accounting occurs

Then bytes are pending reclaim rather than free.

### Requirement: Pool Drain Is Safe

Draining pool SHALL stop new normal allocations while allowing existing leases
to complete.

#### Scenario: Model migration

Given old Device pool is draining

When in-flight work finishes

Then leases can retire before pool closes.