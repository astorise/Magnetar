# Define Memory Planning Model

## Why

Magnetar represents tensor storage as opaque Provider-owned resources.

Compute Graph submission, Tensor Descriptors, Resource Affinity, explicit Data
Movement and Provider Compute Advertisements now define the portable boundary of
`magnetar:compute/run`.

The Runtime now needs a Memory Planning Model.

Memory planning is a native Runtime responsibility.

It determines how tensor resources, intermediate values, materialized views,
temporary buffers, transfers and outputs are placed, reused and released during
execution.

Memory planning must not expose allocators, raw buffers, backend storage, GPU
pointers, native queues or Provider handles through WIT.

Components describe compute work.

Providers execute native work.

The Runtime plans resource lifetimes, placement constraints and memory pressure
before execution.

## What Changes

This proposal introduces the Memory Planning Model.

The Memory Planning Model defines:

- memory requirements
- tensor resource lifetimes
- intermediate value lifetimes
- temporary buffer requirements
- materialization requirements
- transfer buffer requirements
- output allocation requirements
- memory pressure handling
- Provider and Device memory constraints
- memory planning diagnostics

The model uses:

- Tensor Descriptors
- Resource Affinity
- Provider Compute Advertisements
- Data Movement operations
- Compute Graph structure

This proposal does not expose raw memory allocation to Components.

This proposal does not introduce a WIT memory allocation Capability.

This proposal does not define Provider-specific memory pools.

This proposal does not define live resource migration.

## Impact

The Runtime can validate memory feasibility before Provider execution.

Providers can receive a planned execution request with explicit resource
requirements.

Future Scheduler work can reason about memory pressure, placement, reuse,
materialization and transfer cost.

Future Providers can optimize native allocation internally while preserving the
portable `magnetar:compute/run` contract.