## ADDED Requirements
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
