## ADDED Requirements

### Requirement: Reference CPU Uses Host Contiguous Tensor Resources

Reference CPU Provider SHALL support host contiguous Tensor Resources for the first operator implementation scope.

#### Scenario: CPU matmul input

Given CPU matmul receives host contiguous f32 tensors

When dispatch runs

Then validation succeeds if shapes are compatible.

---

### Requirement: Reference CPU Rejects Unsupported Tensor Layouts

Reference CPU Provider SHALL reject unsupported layouts unless Runtime plans an explicit conversion.

#### Scenario: Blocked input

Given CPU Kernel receives blocked layout

When no conversion is planned

Then dispatch fails with layout unsupported.

---

### Requirement: Reference CPU Rejects Unsupported Tensor Memory Classes

Reference CPU Provider SHALL reject unsupported memory classes unless Runtime plans explicit movement.

#### Scenario: Device memory input

Given CPU Kernel receives Device-only memory

When no host movement is planned

Then dispatch fails with memory class unsupported.