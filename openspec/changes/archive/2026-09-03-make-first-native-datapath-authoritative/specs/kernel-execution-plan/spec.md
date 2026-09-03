## ADDED Requirements

### Requirement: Prepared Plan Drives Execution
A ready PreparedExecutionPlan SHALL drive first-native dispatch through immutable PlanNodeBinding and PreparedKernelId entries without normal hot-path kernel rediscovery.

#### Scenario: Registry preference changes after publication
- **WHEN** a ready plan is executed after registry preferences change
- **THEN** execution uses the published binding from that plan generation.

#### Scenario: Bound kernel is unavailable
- **WHEN** a plan binding references a missing or revoked PreparedKernelId
- **THEN** Runtime fails with a structured plan execution error.
