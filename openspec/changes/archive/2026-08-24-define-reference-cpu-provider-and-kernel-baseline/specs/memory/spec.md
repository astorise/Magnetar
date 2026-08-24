## ADDED Requirements

### Requirement: Host Memory Supports Reference CPU Kernels

Memory Manager SHALL support host memory resources usable by Reference CPU
Kernels.

#### Scenario: CPU input tensor

Given input tensor is host-resident

When CPU Kernel dispatch runs

Then Memory Manager provides Runtime resource references for host access.

---

### Requirement: CPU Outputs Are Tracked

Outputs produced by Reference CPU Kernels SHALL be tracked by Memory Manager.

#### Scenario: CPU output

Given CPU matmul writes output tensor

When dispatch completes

Then Memory Manager marks output ready and host-resident.

---

### Requirement: CPU Fallback Requires Explicit Movement

If fallback to Reference CPU requires moving data to host memory, movement SHALL
be explicit and policy-controlled.

#### Scenario: Device tensor fallback

Given tensor is Device-resident

When CPU fallback is considered

Then Runtime plans explicit movement or rejects fallback.