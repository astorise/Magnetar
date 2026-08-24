## ADDED Requirements
### Requirement: Reference CPU Covers Required-Now Operators

Reference CPU Provider SHALL provide or explicitly fail validation for
required-now operators.

#### Scenario: Required kernel missing

Given `softmax` is required-now

And no CPU softmax kernel is advertised

When first scope validation runs

Then Runtime reports first-scope-kernel-missing.

---

### Requirement: Reference CPU Does Not Advertise Placeholders As Implemented

Reference CPU Provider SHALL not advertise placeholder operators as implemented
unless they are truly implemented.

#### Scenario: Placeholder advertised falsely

Given `paged-attention` is placeholder

When CPU Provider advertises it as implemented

Then Runtime validates metadata and conformance before accepting it.

---

### Requirement: Reference CPU Uses F32 Baseline

Reference CPU Provider SHALL prioritize f32 compute for the first operator
scope.

#### Scenario: F32 matmul

Given f32 matmul graph

When CPU Provider dispatches it

Then dispatch uses f32-compatible baseline kernel.
