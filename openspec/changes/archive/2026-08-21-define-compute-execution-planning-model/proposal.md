# Define Compute Execution Planning Model

## Why

Magnetar now has the foundational pieces required to describe portable compute
execution:

- Compute Graph submission
- Tensor Descriptors
- Tensor Resources
- explicit Data Movement
- Resource Affinity
- Resolution Policies
- Provider Compute Advertisements
- Memory Planning
- Compute Error Model

However, these pieces must be combined into a coherent execution decision before
work is submitted to a Provider.

The Runtime needs an Execution Planning Model.

Execution Planning is the Runtime step that transforms a validated Compute Graph
and its resources into an Execution Plan.

The Execution Plan records:

- selected Provider
- selected Device
- selected Capability implementation
- required Tensor Resources
- required Data Movement operations
- required Materialization operations
- Memory Plan
- Resource Affinity bindings
- Provider constraints
- expected execution phases
- validation diagnostics

This change does not introduce a Scheduler.

This change does not execute work.

This change does not introduce automatic failover or live migration.

It defines the planning boundary that the future Scheduler will consume.

## What Changes

This proposal introduces the Compute Execution Planning Model.

The Runtime SHALL create an Execution Plan before submitting compute work to a
Provider.

The Execution Plan is produced from:

- Compute Graph
- Tensor Descriptors
- Tensor Resources
- Resource Affinity metadata
- Provider Compute Advertisements
- Resolution Policy
- Data Movement requirements
- Memory Plan

The Execution Plan is used to validate that the selected Provider and Device can
execute the requested graph without violating affinity, memory, layout, dtype,
operation, or placement constraints.

The Execution Plan is a Runtime-native object.

Components do not build Provider-specific plans.

Providers do not choose themselves.

The Scheduler will later use Execution Plans to decide when and where planned
work runs.

## Impact

Compute execution becomes explicit, inspectable and deterministic before
Provider submission.

The Runtime gains a clear boundary between validation, planning, scheduling and
execution.

Future Scheduler work can consume Execution Plans instead of re-deriving
Provider, Device, memory and affinity decisions.

Future Providers can receive validated execution requests with clear constraints
without exposing native implementation details.