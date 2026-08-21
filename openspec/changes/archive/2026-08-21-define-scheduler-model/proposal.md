# Define Scheduler Model

## Why

Magnetar now has a Compute Execution Planning Model.

Execution Planning decides what should run:

- selected Provider
- selected Device
- selected Capability implementation
- Resource Affinity bindings
- Memory Plan
- Data Movement steps
- Materialization steps
- Provider constraints

The Scheduler decides when and how planned work is submitted.

The Scheduler is a Runtime-native responsibility.

It manages queued execution plans, ordering, cancellation, completion,
interruption, backpressure and runtime-visible execution state.

The Scheduler does not replace the Resolution Policy.

The Scheduler does not choose Providers from scratch.

The Scheduler consumes validated Execution Plans produced by the Runtime.

This change defines the Scheduler Model without introducing distributed
execution, live migration or automatic failover.

## What Changes

This proposal introduces the Scheduler Model.

The Scheduler accepts Compute Execution Plans and turns them into scheduled
runtime operations.

The Scheduler manages:

- submission queue
- operation lifecycle
- execution state
- cancellation
- completion
- interruption
- timeout handling
- backpressure
- Provider availability checks
- Device availability checks
- scheduling diagnostics

The Scheduler preserves:

- Resource Affinity
- Provider binding
- Device binding
- Memory Plan
- Execution Plan constraints

This proposal does not define live state migration.

This proposal does not define distributed scheduling.

This proposal does not silently retry Provider-pinned work on another Provider.

This proposal does not expose native queues, streams, threads or handles to
Components.

## Impact

Magnetar gains a clear boundary between planning and execution.

Compute work can be queued, observed, cancelled and completed through stable
runtime state.

Future execution health, retry, batching, distributed scheduling and cost-aware
placement can build on this Scheduler Model without changing the portable
Component contract.