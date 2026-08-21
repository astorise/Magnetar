# Tasks

## Scheduler Types

- [x] Define Scheduler
- [x] Define ScheduledOperation
- [x] Define ScheduledOperationId
- [x] Define SchedulerQueue
- [x] Define SchedulingPolicy
- [x] Define SchedulingState
- [x] Define SchedulingDiagnostic

## Operation Lifecycle

- [x] Define accepted state
- [x] Define queued state
- [x] Define ready state
- [x] Define submitted state
- [x] Define running state
- [x] Define completed state
- [x] Define cancelled state
- [x] Define failed state
- [x] Define interrupted state

## Queue Management

- [x] Accept validated Compute Execution Plans
- [x] Reject invalid Execution Plans
- [x] Preserve Execution Plan constraints
- [x] Preserve Resource Affinity
- [x] Preserve Provider and Device binding
- [x] Enforce queue ordering policy

## Scheduling Policy

- [x] Define deterministic FIFO policy
- [x] Define priority policy placeholder
- [x] Define deadline policy placeholder
- [x] Define resource-aware policy placeholder
- [x] Define batch-aware policy placeholder
- [x] Define fairness policy placeholder

## Provider and Device Checks

- [x] Check Provider availability before submission
- [x] Check Device availability before submission
- [x] Check Resource Affinity before submission
- [x] Check Memory Plan before submission
- [x] Reject Provider-pinned implicit migration

## Cancellation

- [x] Support cancellation before Provider submission
- [x] Support cancellation after Provider submission when safe
- [x] Define cancellation terminal state
- [x] Define cancellation diagnostics
- [x] Define Provider cancellation forwarding

## Completion and Observation

- [x] Expose operation state
- [x] Expose terminal result
- [x] Expose stable execution diagnostics
- [x] Expose Provider and Device stable identifiers
- [x] Hide native handles and backend internals

## Backpressure

- [x] Define queue capacity behavior
- [x] Define operation admission failure
- [x] Define backpressure error
- [x] Define bounded observation behavior

## Interruption

- [x] Define Provider interruption behavior
- [x] Define Device interruption behavior
- [x] Define Runtime interruption behavior
- [x] Preserve Provider-pinned semantics
- [x] Return structured interruption errors

## Documentation

- [x] Document Scheduler lifecycle
- [x] Document relationship with Execution Planning
- [x] Document relationship with Resolution Policy
- [x] Document relationship with Provider execution
- [x] Document cancellation and interruption semantics
- [x] Document no live migration guarantee
