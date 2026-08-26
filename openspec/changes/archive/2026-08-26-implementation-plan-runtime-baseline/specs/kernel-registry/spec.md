## ADDED Requirements

### Requirement: Kernel Registry Precedes E2E Execution

Kernel Registry and Dispatch SHALL be implemented before E2E local inference
success path.

#### Scenario: E2E matmul

Given E2E graph contains matmul

When execution runs

Then Kernel Registry selects an eligible Reference CPU Kernel.

---

### Requirement: Kernel Dispatch Revalidation Included In Baseline

Kernel Dispatch baseline SHALL revalidate Provider, Device, Memory, Resource
Affinity, and policy before dispatch.

#### Scenario: Provider unavailable

Given selected Provider becomes unavailable

When dispatch begins

Then dispatch fails closed or replans according to policy.