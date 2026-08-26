## ADDED Requirements

### Requirement: E2E Uses Kernel Registry

E2E conformance SHALL validate Kernel Registry candidate lookup and selection
for required operators.

#### Scenario: Matmul kernel selected

Given graph contains matmul

When execution is planned

Then Kernel Registry selects an eligible Reference CPU matmul Kernel.

---

### Requirement: E2E Detects Missing Kernels

E2E conformance SHALL include missing kernel failure cases.

#### Scenario: Missing attention kernel

Given attention Operator has no eligible Kernel

When E2E graph execution is planned

Then Runtime reports structured missing Kernel error.