# Tasks

## Provider Execution Types

- [x] Define ProviderExecutionApi
- [x] Define ProviderExecutionHandle
- [x] Define ProviderExecutionRequest
- [x] Define ProviderExecutionStatus
- [x] Define ProviderExecutionResult
- [x] Define ProviderExecutionError
- [x] Define ProviderExecutionDiagnostic

## Preparation

- [x] Accept validated ComputeExecutionPlan
- [x] Validate Provider ownership
- [x] Validate selected Device binding
- [x] Validate Resource Affinity binding
- [x] Validate Memory Plan compatibility
- [x] Validate Data Movement steps
- [x] Reject unplanned work

## Submission

- [x] Submit planned work to Provider
- [x] Return ProviderExecutionHandle
- [x] Preserve ScheduledOperation identity
- [x] Preserve ExecutionPlan constraints
- [x] Prevent Provider-side re-resolution

## Observation

- [x] Query Provider execution status
- [x] Map Provider status to ScheduledOperation state
- [x] Report stable progress metadata when available
- [x] Prevent exposure of native queues, streams or handles

## Cancellation

- [x] Forward cancellation request to Provider
- [x] Handle unsupported cancellation
- [x] Handle cancellation accepted
- [x] Handle cancellation race with completion
- [x] Map cancellation result to stable terminal state

## Completion

- [x] Collect execution result
- [x] Return produced TensorResources
- [x] Attach Resource Affinity to produced resources
- [x] Release temporary Provider-owned execution resources
- [x] Preserve terminal state

## Error Mapping

- [x] Map Provider validation errors
- [x] Map Provider submission errors
- [x] Map Provider execution errors
- [x] Map Provider interruption errors
- [x] Map Provider cancellation errors
- [x] Attach diagnostics without exposing native internals

## Lifecycle

- [x] Define prepare lifecycle
- [x] Define submit lifecycle
- [x] Define observe lifecycle
- [x] Define cancel lifecycle
- [x] Define complete lifecycle
- [x] Define release lifecycle

## Documentation

- [x] Document Runtime-to-Provider execution boundary
- [x] Document why this is not a WIT Capability
- [x] Document relationship with Scheduler
- [x] Document relationship with Execution Plan
- [x] Document Provider privacy rules
- [x] Document no automatic failover guarantee
