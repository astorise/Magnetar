## ADDED Requirements

### Requirement: Memory Manager Owns Tensor Resource Allocation

Memory Manager SHALL allocate, track, admit, release, evict, and invalidate Tensor Resources.

#### Scenario: Allocate tensor

Given Runtime plans output tensor

When Memory Manager admits allocation

Then Tensor Resource metadata is created or updated.

---

### Requirement: Memory Manager Tracks Tensor Residency

Memory Manager SHALL track Tensor Resource residency, memory class, host visibility, Provider/Device affinity, transfer state, conversion state, and eviction eligibility.

#### Scenario: Tensor transfer

Given Tensor moves from Device to host

When transfer completes

Then Memory Manager updates residency and Resource Affinity metadata.

---

### Requirement: Memory Manager Validates Tensor Size

Memory Manager SHALL compute or conservatively estimate Tensor Resource size.

#### Scenario: Unknown packed size

Given packed quantized tensor size cannot be computed

When admission runs

Then Memory Manager rejects or applies conservative policy.

---

### Requirement: Memory Manager Tracks Tensor Views

Memory Manager SHALL track Tensor View lifetime dependency on base resources.

#### Scenario: Base tensor evicted

Given a view depends on base tensor

When base tensor is evicted

Then the view becomes invalid or unavailable.

---

### Requirement: Memory Manager Enforces Tensor Mutability And Aliasing

Memory Manager SHALL participate in mutability and aliasing validation where storage ownership is affected.

#### Scenario: Immutable tensor mutation

Given Tensor Resource is immutable

When Kernel requests write access

Then Memory Manager rejects or reports mutability violation.