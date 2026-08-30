## ADDED Requirements
### Requirement: Tensor Resource Uses Logical Allocation Backing

Tensor Resource SHALL reference an AllocationLease rather than owning a dedicated
native allocation.

#### Scenario: Intermediate Tensor

Given Tensor needs 8 MiB transient Device storage

When created

Then it SHALL bind to sub-region of transient DeviceMemoryPool.

### Requirement: Tensor Logical Size Is Independent From Allocator Padding

Allocator alignment/padding SHALL not change Tensor byte-length semantics.

#### Scenario: 1000-byte Tensor in 1024-byte slot

Given allocator rounds storage upward

When Tensor descriptor is inspected

Then logical payload remains 1000 bytes.

### Requirement: Tensor Reuse Does Not Merge Resource Identity

Two Tensors using same physical bytes at different times SHALL remain distinct
logical Tensor Resources.

#### Scenario: A then C reuse slot

Given C reuses A's released storage

When diagnostics inspect Resources

Then A and C retain distinct Resource IDs.

### Requirement: Tensor Views Preserve Underlying Lease Lifetime

A live ResourceView SHALL keep the underlying AllocationLease valid.

#### Scenario: Parent Tensor released

Given View remains live

When lease reclamation runs

Then backing remains allocated.

### Requirement: Tensor Alignment Is Validated Against Planned Slot

A Tensor SHALL not bind to a slot violating its required alignment.

#### Scenario: Kernel needs 256-byte alignment

Given candidate slot only guarantees 64 bytes

When binding occurs

Then slot is rejected.