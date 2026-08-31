## ADDED Requirements

### Requirement: First-Native Generation Requires Prepared Plans
First-native model execution SHALL use a PreparedExecutionPlan compatible with the ModelInstance, phase, Provider, Device, dtype, layout, KV mode, and workload bucket.

#### Scenario: Compatible plan selected
- **WHEN** Runtime starts first-native prefill or decode
- **THEN** Runtime selects or builds a PreparedExecutionPlan whose guards match the current execution request.

#### Scenario: Missing plan fails closed
- **WHEN** no compatible PreparedExecutionPlan is available for first-native execution
- **THEN** Runtime rejects generation with a structured plan-unavailable or graph-planning error.

#### Scenario: Invalidated plan rejected
- **WHEN** a PreparedExecutionPlan has been invalidated
- **THEN** Runtime MUST NOT use it to produce logits for new work.

### Requirement: First-Native Evidence Identifies Plan Generation
First-native execution evidence SHALL identify the actual PreparedExecutionPlan generation used for each prefill and decode step.

#### Scenario: Plan generation observed
- **WHEN** a first-native execution step completes
- **THEN** observations include the PlanId or equivalent opaque identity and the PlanGeneration used by that step.
