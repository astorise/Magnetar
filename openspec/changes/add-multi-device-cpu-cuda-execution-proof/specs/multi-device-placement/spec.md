## ADDED Requirements

### Requirement: Runtime Supports Concurrent Multi-Provider Registration

Runtime SHALL support registering more than one Provider simultaneously, with every registered Provider's Kernels present in one shared Kernel Registry.

#### Scenario: Two heterogeneous Providers registered together

- **GIVEN** a Reference CPU Provider and a CUDA Provider, each exposing its own real Device
- **WHEN** both are registered into the same Runtime
- **THEN** Runtime construction succeeds
- **AND** both Providers' Kernels are present in that Runtime's Kernel Registry

### Requirement: Kernel Registry Candidate Selection Is Provider-Ranked, Not Provider-Filtered By Affinity

Kernel Registry candidate selection SHALL rank every structurally compatible candidate across every registered Provider by its existing cost/pressure/fallback-rank policy. It SHALL NOT exclude a candidate solely because its Provider differs from the requesting `ResourceAffinity`'s own Provider. A caller requiring a specific Provider SHALL select that Provider's own candidate explicitly from the full candidate list rather than relying on the highest-ranked selection alone.

#### Scenario: Two Providers advertise the same Operator

- **GIVEN** two registered Providers that both advertise a Kernel for the same Operator
- **AND** a `KernelSelectionRequest` whose `ResourceAffinity` names one specific Provider
- **WHEN** candidate selection ranks the results
- **THEN** the highest-ranked candidate is not guaranteed to belong to the Provider named by the request's `ResourceAffinity`
- **AND** a caller that requires that specific Provider's Kernel obtains it by filtering the full candidate list for that Provider, not by trusting the top-ranked selection

### Requirement: A Multi-Device Placement Plan Can Be Built From a Real Execution's Own Data

`MultiDevicePlacementPlan`, `DeviceSet`, `PipelineStage`, and `StageMovementEdge` SHALL be constructible from the real `DeviceMetadata`, `DeviceAvailability`, and `TensorResourceId`s of an execution that has already run, as an explicit, structurally-validated record of that execution's real cross-Device stages and movement.

#### Scenario: Two real Devices, two real stages

- **GIVEN** two real Devices from two real Providers, and a real two-stage computation that has already executed successfully across both
- **WHEN** a `DeviceSet` is built from both Devices' own real metadata/availability, and a `MultiDevicePlacementPlan` is built with one `PipelineStage` per real stage and one `StageMovementEdge` for the real cross-Device movement between them
- **THEN** the Plan reaches `Ready` via `mark_ready`
- **AND** its fingerprint is real and stable

### Requirement: Multi-Device Placement Plan Does Not Yet Gate Execution

A `MultiDevicePlacementPlan` SHALL be permitted to exist purely as a descriptive record: no execution path in this Runtime is required to consult or be gated by one until a future change wires it in as an enforced contract.

#### Scenario: Execution proceeds without consulting a Plan

- **GIVEN** a real cross-Device execution driven directly through `KernelSelectionRequest`/`KernelDispatchPlan`/`KernelDispatcher`
- **WHEN** a `MultiDevicePlacementPlan` is built afterward to describe what happened
- **THEN** the execution's own success or failure did not depend on that Plan's existence, content, or state
