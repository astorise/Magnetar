## ADDED Requirements
### Requirement: Device Compatibility Constrains Kernels

Kernel execution SHALL validate Device compatibility.

#### Scenario: Kernel requires CUDA capability

Given a Kernel requires a CUDA-capable Device

When Runtime plans execution on CPU Device

Then Runtime rejects the Kernel.

---

### Requirement: Device State Affects Kernel Dispatch

Device readiness, memory pressure, loss, reset, or unavailability SHALL affect
Kernel dispatch eligibility.

#### Scenario: Device unavailable

Given a Device is unavailable

When Runtime considers a Device-bound Kernel

Then the Kernel is not eligible.

---

### Requirement: Device Metadata Supports Kernel Planning

Device metadata SHALL expose features needed for Kernel planning, such as
memory class support, dtype support, layout support, execution limits, and
hardware feature flags.

#### Scenario: Tensor core requirement

Given a Kernel requires tensor-core-like capability

When Runtime validates Device metadata

Then the Device must advertise compatible feature metadata.
