# Define Compute Data Movement Model

## Why

Magnetar represents tensor storage as opaque Provider-owned resources.

Once compute graphs can consume and produce Tensor Resources, the Runtime needs
explicit rules for moving data between host memory, Providers, Devices and
materialized tensor resources.

Data movement must not be implicit.

The Runtime must not silently stage through CPU memory, copy across Devices,
materialize views, or introduce synchronization barriers without the operation
being represented in the compute model.

This change defines the portable data movement model for `magnetar:compute/run`.

## What Changes

This proposal introduces explicit data movement operations.

The model includes:

- upload
- download
- copy
- materialize
- transfer
- dtype conversion
- placement conversion

The model distinguishes:

- host-owned data
- Provider-owned Tensor Resources
- Device-bound Tensor Resources
- tensor views
- materialized tensor copies

All data movement operations must validate:

- tensor descriptors
- resource affinity
- Provider compatibility
- Device compatibility
- dtype support
- layout support
- size limits

This proposal does not expose raw host pointers, GPU pointers, native buffers,
backend storage, queues, streams or Provider handles.

This proposal does not introduce implicit fallback or live resource migration.

## Impact

Tensor movement becomes explicit and auditable.

The Runtime can reject incompatible Provider or Device usage before execution.

Future Scheduler and memory planner changes can reason about transfer cost,
placement, materialization and locality.

Providers can advertise data movement support separately from compute operation
support.