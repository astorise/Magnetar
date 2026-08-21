# Tasks

## Execution Plan Types

- [x] Define ComputeExecutionPlan
- [x] Define ExecutionPlanId
- [x] Define ExecutionStep
- [x] Define ExecutionPhase
- [x] Define ExecutionInput
- [x] Define ExecutionOutput
- [x] Define ExecutionConstraint
- [x] Define ExecutionDiagnostic

## Planning Inputs

- [x] Use Compute Graph as planning input
- [x] Use Tensor Descriptors as planning input
- [x] Use Tensor Resources as planning input
- [x] Use Resource Affinity as planning input
- [x] Use Provider Compute Advertisement as planning input
- [x] Use Resolution Policy as planning input
- [x] Use Memory Plan as planning input
- [x] Use Data Movement requirements as planning input

## Provider and Device Selection

- [x] Select Provider through Resolution Policy
- [x] Select Device through Provider advertisement and resource constraints
- [x] Validate Capability version compatibility
- [x] Validate operation schema support
- [x] Validate dtype support
- [x] Validate layout support
- [x] Validate precision policy support
- [x] Validate deterministic behavior requirements

## Resource and Affinity Planning

- [x] Bind input Tensor Resources to the plan
- [x] Bind output Tensor Resources to the plan
- [x] Preserve Provider-pinned affinity
- [x] Preserve Device-bound affinity
- [x] Preserve Affinity Group constraints
- [x] Reject incompatible resource chains
- [x] Record required explicit transfers
- [x] Record required explicit materialization

## Memory Planning Integration

- [x] Attach Memory Plan to Execution Plan
- [x] Validate peak memory requirements
- [x] Validate temporary buffer requirements
- [x] Validate transfer buffer requirements
- [x] Validate materialization memory requirements
- [x] Validate output allocation requirements

## Data Movement Integration

- [x] Include Upload steps when required
- [x] Include Download steps when required
- [x] Include Copy steps when required
- [x] Include Transfer steps when required
- [x] Include Materialize steps when required
- [x] Prevent hidden CPU staging

## Runtime Validation

- [x] Validate Execution Plan before scheduling
- [x] Validate that all dependencies are resolved coherently
- [x] Validate that no implicit Provider migration is planned
- [x] Validate that all required resources are available
- [x] Return structured execution-planning errors

## Documentation

- [x] Document planning lifecycle
- [x] Document relationship with Resolution Policy
- [x] Document relationship with Memory Planning
- [x] Document relationship with Scheduler
- [x] Document examples for CPU/GPU Provider selection
- [x] Document examples for Provider-pinned resources