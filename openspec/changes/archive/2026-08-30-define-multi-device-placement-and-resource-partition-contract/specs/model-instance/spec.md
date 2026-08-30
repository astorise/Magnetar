## ADDED Requirements
### Requirement: Model Instance May Use Multiple Devices

One Model Instance SHALL be able to own execution state across multiple local Devices.

#### Scenario: Large model

Given weights cannot fit on one GPU

When Model Instance loads

Then Runtime may create multi-Device placement and memory state.

### Requirement: Model Instance Tracks Placement Generation

Model Instance SHALL identify active MultiDevicePlacementPlan generation.

#### Scenario: Re-placement

Given replacement Plan becomes active

When diagnostics inspect Model Instance

Then new placement generation is distinguishable.

### Requirement: Model Readiness Requires Mandatory Devices

If placement requires multiple Devices, Model Instance SHALL not become READY
until all mandatory placement dependencies are ready.

#### Scenario: GPU1 kernels unprepared

Given GPU0 is ready but GPU1 mandatory stage is not

When readiness is evaluated

Then Model Instance is not fully ready under that Plan.

### Requirement: Model Instance May Own Degraded Plans

Model Instance SHALL be able to retain pre-built fallback placement plans.

#### Scenario: GPU1 failure

Given validated GPU0-only degraded Plan exists

When GPU1 fails

Then Runtime may activate degraded Plan according to policy.

### Requirement: Model Revision Invalidates Placement As Needed

Changes to graph/resource requirements SHALL be able to invalidate current placement.

#### Scenario: Adapter revision increases memory demand

Given new adapter no longer fits current Device allocation

When revision changes

Then placement is rebuilt or rejected.
