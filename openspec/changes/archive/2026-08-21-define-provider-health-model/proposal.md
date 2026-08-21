# Define Provider Health Model

## Why

Magnetar Providers execute native work on Devices.

The Runtime needs a stable way to observe Provider and Device availability
before resolution, scheduling and execution.

Provider availability affects:

- Capability resolution
- Resolution Policy decisions
- Scheduler admission
- Provider execution submission
- interruption reporting
- diagnostics
- backpressure

However, Provider health is not automatic failover.

A healthy Provider may still fail during execution.

An unhealthy Provider may be rejected before submission.

A Provider becoming unavailable after state creation does not allow the Runtime
to silently migrate Provider-pinned resources to another Provider.

The Runtime therefore needs a Provider Health Model that reports availability
and capacity without implying live migration, automatic retry or execution
failover.

## What Changes

This proposal introduces the Provider Health Model.

The model defines health reporting for:

- Providers
- Devices
- Capability implementations
- execution contexts
- Scheduler-visible capacity

The model introduces stable health states:

- unknown
- initializing
- available
- degraded
- saturated
- draining
- unavailable
- interrupted

The Runtime uses health information during:

- Capability resolution
- Resolution Policy evaluation
- Execution Planning
- Scheduler admission
- Provider Execution API submission
- interruption diagnostics

This proposal does not introduce automatic failover.

This proposal does not introduce live resource migration.

This proposal does not introduce distributed health monitoring.

This proposal does not expose native handles, driver objects, queues, streams or
backend-private diagnostics to Components.

## Impact

Provider and Device availability become explicit Runtime concepts.

Resolution Policies can reject unavailable or degraded Providers.

The Scheduler can avoid submitting work to unavailable Providers or Devices.

Execution failures caused by Provider or Device loss can be reported as
interruptions.

Future retry, replanning, batching and distributed scheduling work can build on
this health model without changing the Provider execution boundary.