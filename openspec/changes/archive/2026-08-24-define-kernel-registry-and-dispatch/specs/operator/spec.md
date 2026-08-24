## ADDED Requirements
### Requirement: Operator Invocation Drives Kernel Selection

Operator invocation metadata SHALL drive Kernel Registry candidate lookup and
selection.

#### Scenario: Matmul invocation

Given graph planning produces a matmul Operator invocation

When Kernel Registry is queried

Then candidate Kernels implementing matmul are considered.

---

### Requirement: Operator Semantics Constrain Dispatch

Kernel Dispatch SHALL preserve Operator semantics.

#### Scenario: Fused candidate

Given a fused Kernel is selected

When dispatch is planned

Then Runtime validates that Operator semantics remain preserved.