# Scheduler Model

The Scheduler is the Runtime-owned boundary between validated
`ComputeExecutionPlan` values and Provider submission. It accepts planned work,
places it in a bounded queue, exposes stable operation state, and preserves the
Provider, Device, Resource Affinity and Memory Plan decisions made during
execution planning.

Components never receive native Provider queues, Device streams, thread handles,
locks, GPU pointers or backend execution handles. They observe scheduled work
through stable `ScheduledOperationId`, `SchedulingState`,
`ScheduledOperationResult` and `SchedulingDiagnostic` values.

## Lifecycle

1. The Runtime validates a `ComputeExecutionPlan`.
2. The Scheduler accepts only validated plans.
3. Accepted work is queued according to the active `SchedulingPolicy`.
4. The initial policy is deterministic FIFO.
5. Before Provider submission, the Scheduler checks selected Provider and Device
   availability when that information is available.
6. The operation moves through accepted, queued, ready, submitted, running and a
   terminal state.
7. Completed, cancelled, failed and interrupted operations remain terminal.

## Queueing And Backpressure

`SchedulerQueue` is bounded. When capacity is exhausted, admission fails with a
structured `QueueCapacityExceeded` scheduler error. Bounded observation and
stable operation identifiers avoid exposing native queues or backend runtime
internals.

FIFO ordering is deterministic: operations accepted first are selected first.
Priority, deadline, resource-aware, batch-aware and fairness policies are
defined as placeholders for later policy implementations. These policies operate
on already planned work and do not replace Resolution Policy.

## Cancellation

Queued work can be cancelled before Provider submission. It reaches the
cancelled terminal state without invoking the Provider.

After Provider submission, cancellation is forwarded only when the selected
Provider can safely support it. The current model records the cancellation
request and returns a structured unsupported-cancellation error rather than
pretending cancellation succeeded.

## Interruption

Provider, Device or Runtime unavailability during scheduling is reported as an
interrupted terminal state. Provider-pinned work is not silently migrated to
another Provider after state creation or observable output. A future retry or
replanning policy may use restartability metadata, but automatic replay is not
part of this Scheduler model.

## Relationship To Execution Planning

Execution Planning decides what should run. Scheduling decides when planned work
runs. The Scheduler must not rewrite the selected Provider, selected Device,
Resource Affinity, Memory Plan, data movement or materialization steps without a
new validated execution plan.

## Relationship To Provider Execution

Providers own native execution details internally. Scheduler diagnostics may
include stable Provider and Device identifiers, queue order, terminal state and
structured failure reasons, but not native handles, queues, streams, locks, GPU
pointers, backend storage, credentials or ambient filesystem paths.
