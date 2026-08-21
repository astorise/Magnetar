## ADDED Requirements

### Requirement: Compute Execution Planning

The Runtime SHALL define a Compute Execution Planning Model.

Execution Planning SHALL transform a validated Compute Graph and its resources
into a Compute Execution Plan.

#### Scenario: Create execution plan

Given a validated Compute Graph

When the Runtime prepares execution

Then it creates a Compute Execution Plan before scheduling or Provider
submission.

---

### Requirement: Execution Plan

A Compute Execution Plan SHALL describe how compute work is intended to execute.

The plan SHALL include:

- selected Provider
- selected Device when applicable
- selected Capability implementation
- Compute Graph reference
- input resources
- output descriptors
- required data movement
- required materialization
- Memory Plan
- Resource Affinity bindings
- validation diagnostics

#### Scenario: Inspect execution plan

Given a Compute Execution Plan

When diagnostics are requested

Then the Runtime reports stable identifiers, constraints and planning decisions
without exposing native handles.

---

### Requirement: Runtime-Owned Plan

A Compute Execution Plan SHALL be owned by the Runtime.

Components SHALL NOT construct Provider-specific execution plans.

Providers SHALL NOT select themselves.

#### Scenario: Component submits compute graph

Given a Component submits a Compute Graph

When execution planning occurs

Then the Runtime selects the Provider and Device according to policy and
constraints.

---

### Requirement: Resolution Policy Integration

Execution Planning SHALL use the active Resolution Policy.

The Resolution Policy SHALL evaluate compatible Providers and Devices using
Capability requirements, Provider advertisements, Resource Affinity and graph
constraints.

#### Scenario: Select Provider

Given multiple Providers implement `magnetar:compute/run`

When Execution Planning evaluates the Compute Graph

Then the Runtime selects a Provider according to the active Resolution Policy.

---

### Requirement: Provider Advertisement Integration

Execution Planning SHALL validate Provider Compute Advertisements.

The selected Provider SHALL advertise support for every required operation
schema, dtype, layout, precision policy, data movement requirement and Device
constraint.

#### Scenario: Provider lacks operation support

Given a Compute Graph requires an operation schema

And the candidate Provider does not advertise support for that schema

When Execution Planning evaluates the candidate

Then the Runtime rejects that Provider for the plan.

---

### Requirement: Resource Affinity Integration

Execution Planning SHALL preserve Resource Affinity.

Provider-pinned and Device-bound resources SHALL NOT be planned for incompatible
Providers or Devices without explicit transfer, copy, materialization, replay,
reload or future migration support.

#### Scenario: Provider-pinned tensor input

Given a Tensor Resource is bound to one Provider

When the Runtime creates an Execution Plan

Then the plan either uses that Provider, includes an explicit supported transfer,
or rejects the plan.

---

### Requirement: Affinity Group Preservation

Execution Planning SHALL preserve Affinity Groups.

Resources belonging to the same Affinity Group SHALL be planned as a coherent
resource chain.

#### Scenario: Coherent model resource chain

Given resources share an Affinity Group

When Execution Planning validates dependent calls

Then the Runtime rejects incompatible Provider, Device, artifact, tokenizer or
template combinations.

---

### Requirement: Memory Plan Integration

A Compute Execution Plan SHALL include or reference a Memory Plan.

The Runtime SHALL validate memory feasibility before scheduling when sufficient
metadata is available.

#### Scenario: Insufficient memory

Given the selected Device cannot satisfy the Memory Plan

When Execution Planning validates the plan

Then the Runtime rejects the plan with a structured memory-planning error.

---

### Requirement: Data Movement Planning

A Compute Execution Plan SHALL include explicit Data Movement steps when data
must move between host memory, Providers, Devices, layouts or materialized
resources.

The Runtime SHALL NOT hide upload, download, copy, transfer, materialization or
host staging.

#### Scenario: Cross-device input

Given a Tensor Resource is bound to one Device

And the selected Provider requires it on another Device

When Execution Planning occurs

Then the plan includes an explicit supported transfer or rejects execution.

---

### Requirement: Execution Materialization Planning

A Compute Execution Plan SHALL include explicit Materialization steps when a
view must become a distinct Tensor Resource.

Materialization SHALL NOT be implicit.

#### Scenario: View unsupported by Provider

Given a Tensor Resource is a view

And the selected Provider cannot consume the view directly

When Execution Planning occurs

Then the plan includes explicit materialization or rejects execution.

---

### Requirement: Execution Phases

A Compute Execution Plan SHALL record expected execution phases.

Execution phases MAY include:

- validation
- resolution
- planning
- data movement
- materialization
- memory allocation
- Provider submission
- execution
- completion
- cancellation
- interruption

#### Scenario: Report phase-specific failure

Given Execution Planning fails during data movement planning

When the Runtime reports the error

Then the error phase identifies the planning step that failed.

---

### Requirement: No Implicit Failover

Execution Planning SHALL NOT imply automatic failover.

The Runtime SHALL NOT plan live migration of Provider-pinned resources unless a
future migration contract explicitly defines it.

#### Scenario: Provider-pinned session

Given work depends on Provider-pinned live state

When another Provider also supports the required Capability

Then Execution Planning does not silently move the live state to that Provider.

---

### Requirement: Restartability Classification

Execution Planning SHALL classify whether planned work is transparent,
restartable or Provider-pinned.

The classification SHALL be based on Resource Affinity, execution phase and
observable output constraints.

#### Scenario: Classify planned compute work

Given a Compute Graph uses only replayable host inputs

When Execution Planning completes before state creation

Then the plan may be classified as transparent or restartable according to the
Resolution Policy.

---

### Requirement: Plan Validation Before Scheduling

The Runtime SHALL validate a Compute Execution Plan before handing it to the
Scheduler.

#### Scenario: Invalid execution plan

Given an Execution Plan has unresolved dependencies

When the Runtime validates it

Then the Runtime rejects the plan before scheduling.

---

### Requirement: Provider-Owned Native Execution

The Compute Execution Plan SHALL NOT expose native execution details.

Native execution details include:

- backend storage
- raw buffers
- GPU pointers
- device queues
- streams
- locks
- kernel symbols
- allocator internals
- Provider handles

#### Scenario: Provider receives planned work

Given the Runtime submits planned work to a Provider

When the Provider executes it

Then the Provider uses native implementation details internally without exposing
them through the portable plan.

---

### Requirement: Execution Planning Diagnostics

Execution Planning SHALL support stable planning diagnostics.

Diagnostics MAY include:

- selected Provider identifier
- selected Device identifier
- selected Capability version
- rejected Provider candidates
- rejected Device candidates
- memory estimates
- transfer requirements
- materialization requirements
- policy decision reasons

Diagnostics SHALL NOT expose native handles, credentials, raw backend errors or
unstable Provider internals as stable contract values.

#### Scenario: Inspect planning diagnostics

Given Execution Planning rejects all Providers

When diagnostics are available

Then the Runtime reports stable candidate identifiers and rejection reasons.

---

### Requirement: Structured Execution Planning Errors

The Runtime SHALL return stable structured errors for Execution Planning
failures.

Structured errors SHALL include categories for:

- planning failed
- no compatible Provider
- no compatible Device
- policy rejected Provider
- unsupported operation
- unsupported dtype
- unsupported layout
- unsupported precision policy
- incompatible resource affinity
- unresolved Affinity Group
- memory plan failed
- data movement required
- unsupported transfer
- materialization required
- Provider unavailable
- Device unavailable

Backend diagnostics MAY be attached for debugging but SHALL NOT define the
stable contract.

#### Scenario: Report planning failure

Given Execution Planning cannot create a valid plan

When the Runtime reports the failure

Then the error uses a stable structured Execution Planning error variant.
