# multi-device-placement Specification

## Purpose
Defines Magnetar's local multi-Device placement contract: Runtime-owned placement decisions, explicit placement plans, eligibility-before-ranking, explicit cross-Device movement, and heterogeneous-Device support -- plus the real, hardware-verified facts this repository has actually confirmed about it: concurrent multi-Provider Runtime registration, the Kernel Registry's real Provider-ranking (not Provider-filtering) candidate selection, a `MultiDevicePlacementPlan` buildable as an explicit record from a real execution's own data (`add-multi-device-cpu-cuda-execution-proof`, CPU+CUDA), and a real chained computation genuinely executing across two physically distinct real GPUs in one Runtime (`add-real-second-gpu-cuda-provider`, verified on `arc-gpu-magnetar`'s CI node). Full production `ModelInstance`-level placement across more than one real GPU remains unimplemented -- `ModelInstancePlacement` still structurally binds one Provider/Device per instance -- and real peer-to-peer GPU-to-GPU movement, per-Device memory feasibility ranking against a genuinely heterogeneous budget, and Device-loss/degraded-replan behavior remain unverified.
## Requirements
### Requirement: Runtime Owns Multi Device Placement

Runtime SHALL own concrete Provider/Device placement decisions.

#### Scenario: Model spans two GPUs

Given Model Component describes portable Transformer graph

When Runtime determines model does not fit optimally on one Device

Then Runtime may place graph segments on GPU0 and GPU1.

### Requirement: Model Component Cannot Select Device

Model Component SHALL NOT authoritatively name concrete Device for execution.

#### Scenario: Portable block

Given Model Component emits block 12

When graph is constructed

Then block semantics do not require `gpu-1`.

### Requirement: Placement Plan Is Explicit

Multi-Device execution SHALL be represented through a
MultiDevicePlacementPlan or equivalent explicit Runtime contract.

#### Scenario: Two-stage pipeline

Given blocks 0..15 execute on GPU0

And blocks 16..31 execute on GPU1

When Plan is inspected

Then both bindings and movement boundary are explicit.

### Requirement: Placement Eligibility Precedes Ranking

Runtime SHALL reject incompatible Device placements before performance ranking.

#### Scenario: Fast Device lacks Kernel

Given GPU1 benchmarks faster

But required Kernel is unavailable

When placement is evaluated

Then GPU1 is excluded for that segment.

### Requirement: Cross Device Movement Is Explicit

A Tensor crossing Device placement boundary SHALL use explicit Resource
movement or explicit peer-access path.

#### Scenario: GPU0 stage feeds GPU1 stage

Given output resides on GPU0

When GPU1 consumes it

Then Runtime represents movement/access dependency explicitly.

### Requirement: Host Staging Policy Survives Multi Device Placement

Multi-Device placement SHALL respect existing host-staging policy.

#### Scenario: Only staged route available

Given GPU0 to GPU1 requires host temporary

And host staging is forbidden

When placement is validated

Then Plan is rejected or another placement is chosen.

### Requirement: Peer Capability Is Explicit

Runtime SHALL not infer peer access from Device similarity.

#### Scenario: Same vendor GPUs

Given two GPUs have no usable peer path

When direct peer access is evaluated

Then zero-copy peer placement is rejected.

### Requirement: Per Device Memory Feasibility

Every Device binding SHALL satisfy its own Memory Manager capacity policy.

#### Scenario: Equal layer split exceeds GPU1 memory

Given GPU0 has 24 GiB and GPU1 has 8 GiB

When placement is planned

Then Runtime does not assume a 50/50 split is feasible.

### Requirement: Heterogeneous Devices Are Supported

MultiDevicePlacementPlan SHALL be able to contain Devices with different capabilities.

#### Scenario: Different GPU generations

Given GPU0 and GPU1 support different Kernel specializations

When Plan is built

Then each stage uses compatible Kernel/Device bindings.

### Requirement: Placement Change Uses New Plan Generation

Concrete placement SHALL not silently mutate under in-flight execution.

#### Scenario: GPU1 becomes preferable

Given active Plan uses GPU0

When Runtime re-places work

Then replacement Plan generation is prepared and safely published.

### Requirement: Device Loss Invalidates Dependent Placement

Hard Device loss SHALL invalidate placement bindings requiring that Device.

#### Scenario: GPU1 reset

Given active Plan requires GPU1

When Device is lost

Then no new work uses that Plan.

### Requirement: Degraded Placement Requires Valid Plan

Runtime SHALL not assume remaining Devices can execute model after Device loss.

#### Scenario: Two-GPU model loses one GPU

Given no validated one-GPU fallback exists

When GPU1 fails

Then Runtime returns structured degraded-plan-unavailable state.

### Requirement: Local Multi Device Scope

Baseline contract SHALL remain local to one Runtime/host.

#### Scenario: Remote node offered

Given Device resides on another host

When this contract evaluates placement

Then remote execution is outside its scope.

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

### Requirement: A Real Chained Computation Can Execute Across Two Physically Distinct Real GPUs

Given two available, distinctly-named CUDA Provider instances bound to two different real GPU ordinals registered into one Runtime, a caller SHALL be able to dispatch a multi-stage computation across both real Devices, with each stage's output explicitly, physically movable to the other Device via an explicit host round trip.

#### Scenario: Two-stage computation across two real GPUs

- **GIVEN** two real, physically distinct GPUs, each with its own registered CUDA Provider instance
- **WHEN** stage one executes on the first real GPU, its result is read back to the host, and admitted fresh into the second real GPU's memory domain for stage two
- **THEN** stage two's real, GPU-computed result matches the expected value for the full chained computation

#### Scenario: Homogeneous real Devices are still tracked as distinct

- **GIVEN** two real GPUs of the identical model and identical memory capacity
- **WHEN** a `DeviceSet` is built from both Devices' own real metadata
- **THEN** the two Devices are recognized as distinct members, not deduplicated by shared architecture/vendor/capacity

