## ADDED Requirements
### Requirement: Tensor Resource May Be Device Resident

Tensor Resource SHALL support residency outside host memory.

#### Scenario: KV Tensor

Given Tensor lives on GPU

When descriptor is inspected

Then logical shape/dtype/layout remain available without copying bytes to host.

### Requirement: Resource View Is Zero Copy By Default

Creating a compatible ResourceView SHALL reference existing storage rather than
copying bytes.

#### Scenario: Slice

Given Tensor has shape `[8, 4096]`

When Runtime creates View of rows 2..4

Then View references same underlying allocation.

### Requirement: View Bounds Are Validated

ResourceView SHALL reject offsets/extents outside underlying storage.

#### Scenario: Overflowed offset

Given offset arithmetic overflows

When View is created

Then operation fails structurally.

### Requirement: View Preserves Residency

A zero-copy View SHALL inherit the residency constraints of underlying storage.

#### Scenario: GPU Tensor View

Given parent Tensor is GPU0-local

When View is created

Then View does not become host-accessible automatically.

### Requirement: Non-Contiguous View Is Explicit

Tensor Resource SHALL describe non-contiguous View through strides/layout when
the View is non-contiguous.

#### Scenario: Transposed View

Given Kernel supports strided input

When View is passed

Then no materialization is required.

#### Scenario: Kernel requires contiguous input

Given Kernel rejects strided input

When Runtime prepares execution

Then it chooses another Kernel or explicit materialization.

### Requirement: View Aliasing Is Preserved

Runtime SHALL know when Views overlap underlying storage.

#### Scenario: Two overlapping slices

Given both reference same allocation region

When asynchronous reads/writes occur

Then hazard tracking treats them as aliases.

### Requirement: Tensor Descriptor Does Not Contain Native Pointer

Logical Tensor descriptor SHALL remain independent from underlying native
address.

#### Scenario: Resource migrates

Given managed memory moves physically

When descriptor is inspected

Then logical Tensor semantics remain unchanged.
