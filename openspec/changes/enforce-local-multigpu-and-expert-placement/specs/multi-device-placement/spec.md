## RENAMED Requirements

- FROM: `### Requirement: Multi-Device Placement Plan Does Not Yet Gate Execution`
- TO: `### Requirement: Production Multi Device Placement Gates Execution`

## MODIFIED Requirements

### Requirement: Production Multi Device Placement Gates Execution

The qualified production multi-GPU profile SHALL require a validated Ready MultiDevicePlacementPlan generation before submitting cross-Device model work. The plan SHALL bind stages, partitions, actual Devices, per-Device reservations and explicitly supported movement/collective edges. Invalidated plans SHALL reject new submissions. Low-level diagnostics MAY retain descriptive after-the-fact plans, but SHALL NOT qualify production model placement; the F32/single-Device baseline remains valid without multi-Device placement.

#### Scenario: Production execution requires a Ready generation

- **WHEN** a cross-Device production model step has a missing, stale or invalidated placement plan
- **THEN** submission is rejected before resource use and an explicit replan is required

#### Scenario: Valid plan controls real transfers

- **GIVEN** distinct real GPUs and a Ready plan with admitted resources and supported movement edges
- **WHEN** a model step executes
- **THEN** stages and movement follow that generation and results match the qualified unsplit reference

#### Scenario: Diagnostic execution remains distinguishable

- **WHEN** low-level tooling constructs a descriptive plan after a direct test dispatch
- **THEN** it remains a diagnostic record and is not evidence that production placement was enforced
