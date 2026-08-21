# Define Provider Compute Advertisement Model

## Why

Magnetar Providers implement Capabilities.

For `magnetar:compute/run`, a Provider may not support every operation schema,
dtype, layout, precision mode, tensor rank, memory size or data movement path.

The Runtime must therefore know what each Provider can execute before selecting
it for a Compute Graph.

Provider selection must not rely only on the existence of the
`magnetar:compute/run` Capability.

It must also consider Provider-advertised compute support.

This change defines how Providers advertise their compute implementation
surface without exposing native handles, backend storage, kernel names, device
queues or hardware-specific APIs.

## What Changes

This proposal introduces the Provider Compute Advertisement Model.

A Provider Compute Advertisement describes:

- supported Compute Capability version
- supported operation schemas
- supported operation families
- supported dtypes
- supported layouts
- supported tensor ranks
- supported shape limits
- supported memory limits
- supported precision modes
- supported deterministic behavior
- supported data movement paths
- supported materialization behavior
- Device-specific constraints
- fallback and recovery classification
- diagnostic metadata

Provider advertisements are used by the Runtime during:

- capability resolution
- Resolution Policy evaluation
- Resource Affinity validation
- Compute Graph validation
- data movement validation
- execution planning

This proposal does not expose native Provider internals.

This proposal does not define live migration.

This proposal does not guarantee that a Provider can execute a graph merely
because it implements `magnetar:compute/run`.

## Impact

Provider selection becomes more precise.

The Runtime can reject unsupported compute graphs before Provider execution.

Components remain portable because they depend on operation schemas and
Capability contracts, not on Provider-specific APIs.

Future Scheduler and memory planner work can use the same advertisements for
placement, cost estimation and execution planning.