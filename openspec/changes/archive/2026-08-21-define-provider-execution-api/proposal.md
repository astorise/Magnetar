# Define Provider Execution API

## Why

Magnetar now has a Scheduler Model.

The Scheduler accepts validated Compute Execution Plans and turns them into
Scheduled Operations.

The Runtime now needs a native API for submitting planned work to Providers.

This API is not a WIT Capability.

It is a Runtime-to-Provider interface used after:

- capability resolution
- resource affinity validation
- provider advertisement validation
- compute graph validation
- memory planning
- execution planning
- scheduling admission

Providers execute native work.

The Runtime owns portable validation, scheduling state and stable error mapping.

Components must never access native Provider execution handles, queues, streams,
threads, backend storage or device-specific APIs.

## What Changes

This proposal introduces the Provider Execution API.

The API allows the Runtime to:

- prepare execution
- submit a validated execution plan
- observe execution state
- request cancellation
- collect completion results
- map Provider errors to stable Runtime errors
- release Provider-owned execution resources

The Provider Execution API operates on Runtime-native objects such as
ComputeExecutionPlan and ScheduledOperation.

Providers may internally use native queues, streams, kernels, allocators,
backend handles and device APIs.

Those native details remain private to the Provider.

This proposal does not expose Provider execution APIs through WIT.

This proposal does not introduce live migration.

This proposal does not introduce automatic retry or failover.

This proposal does not allow Providers to override Runtime planning decisions.

## Impact

The boundary between Scheduler and Provider execution becomes explicit.

Providers receive already validated execution plans.

The Runtime can observe, cancel and complete scheduled work without exposing
native internals to Components.

Future CPU, CUDA, Metal, OpenVINO, Vulkan, QNN or other Providers can implement
the same native execution lifecycle.