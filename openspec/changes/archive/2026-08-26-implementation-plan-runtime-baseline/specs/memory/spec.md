## ADDED Requirements

### Requirement: Memory Baseline Precedes Provider Execution

Memory Manager baseline SHALL be available before Reference CPU Provider writes
Runtime-visible outputs.

#### Scenario: CPU output allocation

Given CPU matmul dispatches

When output is required

Then Memory Manager tracks allocation and readiness.

---

### Requirement: Memory Baseline Tracks Cleanup

Memory baseline SHALL support cleanup sufficient for E2E conformance.

#### Scenario: Session close cleanup

Given E2E session closes

When cleanup runs

Then inference-scoped resources are released or retained only according to
policy.