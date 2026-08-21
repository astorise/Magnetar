# runtime Specification

## Purpose
TBD - created by archiving change bootstrap-runtime. Update Purpose after archive.
## Requirements
### Requirement: Runtime Initialization

The runtime SHALL expose a single initialization entry point.

#### Scenario: Create runtime

Given a valid runtime configuration

When the application creates a runtime

Then a runtime instance is returned.

---

### Requirement: Backend Independence

The runtime SHALL execute independently from any hardware backend.

#### Scenario: No backend implementation

Given a runtime instance

When no backend is registered

Then the runtime initializes successfully.

---

### Requirement: Runtime Lifecycle

The runtime SHALL expose explicit initialization and shutdown phases.

#### Scenario: Shutdown runtime

Given an initialized runtime

When shutdown is requested

Then every registered resource is released.

### Requirement: Memory Planning

The Runtime SHALL define a Memory Planning Model.

Memory planning SHALL be a Runtime responsibility.

Memory planning SHALL NOT be exposed as a direct WIT allocation Capability for
portable Components.

#### Scenario: Plan compute graph memory

Given a Compute Graph

When the Runtime validates it before execution

Then the Runtime creates or validates a Memory Plan for required tensor
resources, intermediate values, temporary buffers and outputs.

---

### Requirement: Opaque Memory

Memory used by Tensor Resources SHALL remain opaque to Components.

Components SHALL NOT receive raw pointers, native buffers, backend storage,
allocators, GPU pointers, queues, streams or Provider handles.

#### Scenario: Component receives tensor output

Given a Compute Graph produces a tensor output

When the output is returned to the Component

Then the Component receives an opaque Tensor Resource and portable metadata only.

---

### Requirement: Tensor Resource Lifetime

The Memory Planning Model SHALL track Tensor Resource lifetimes.

A Tensor Resource SHALL remain live while it may be observed, passed to another
operation, downloaded, transferred, materialized or used by a dependent
resource.

#### Scenario: Prevent premature release

Given a Tensor Resource is still referenced by a Compute Graph output

When the Runtime considers releasing memory

Then the Runtime keeps the resource live.

---

### Requirement: Intermediate Lifetime

The Memory Planning Model SHALL track intermediate value lifetimes inside a
Compute Graph.

Intermediate buffers MAY be reused only after their last required use.

#### Scenario: Reuse intermediate buffer

Given an intermediate tensor is no longer needed

When another compatible intermediate allocation is required

Then the Runtime may reuse the memory only if affinity and layout constraints
permit it.

---

### Requirement: Affinity-Aware Memory Planning

Memory planning SHALL respect Resource Affinity.

Provider-pinned or Device-bound resources SHALL NOT be planned for incompatible
Providers or Devices without an explicit transfer, copy, materialization or
reload step.

#### Scenario: Incompatible memory placement

Given a Tensor Resource is bound to one Device

When a Compute Graph is planned for another Device

Then the Runtime requires explicit data movement before planning execution.

---

### Requirement: Provider Advertisement Awareness

Memory planning SHALL use Provider Compute Advertisements.

Provider advertisements MAY define:

- Device memory limits
- maximum tensor byte size
- maximum element count
- supported layouts
- view consumption support
- materialization requirements
- data movement constraints

#### Scenario: Provider memory limit

Given a Provider advertises a maximum tensor byte size

When the Memory Plan contains a larger tensor

Then the Runtime rejects the plan before Provider execution.

---

### Requirement: Materialization Planning

Memory planning SHALL identify materialization requirements.

A tensor view SHALL NOT be silently materialized unless an explicit
materialization operation exists in the compute or data movement model.

#### Scenario: View requires materialization

Given a Tensor Resource is a view

And the selected Provider cannot consume the view directly

When memory planning occurs

Then the Runtime requires explicit materialization or rejects execution.

---

### Requirement: Data Movement Memory Planning

Memory planning SHALL include memory required for explicit data movement
operations.

Data movement operations include:

- upload
- download
- copy
- materialize
- transfer
- dtype conversion
- layout conversion

#### Scenario: Transfer requires temporary buffer

Given a Transfer operation requires an intermediate staging buffer

When memory planning occurs

Then the Runtime accounts for the staging buffer explicitly.

---

### Requirement: No Hidden CPU Staging

The Runtime SHALL NOT hide CPU staging as an invisible memory planning detail
when it changes placement, cost or synchronization behavior.

#### Scenario: Host-staged transfer

Given a transfer between Providers requires host staging

When the Runtime plans the transfer

Then host staging is represented in the Memory Plan or the transfer is rejected.

---

### Requirement: Memory Feasibility Validation

The Runtime SHALL validate memory feasibility before Provider execution when
sufficient metadata is available.

Validation SHALL consider:

- tensor descriptors
- byte-size calculations
- intermediate lifetimes
- output resources
- Device memory limits
- Provider memory limits
- transfer buffers
- materialization buffers

#### Scenario: Insufficient memory

Given a Compute Graph requires more memory than the selected Device advertises

When the Runtime validates the Memory Plan

Then the Runtime rejects execution with a structured out-of-memory or
memory-planning error.

---

### Requirement: Byte Size Validation

Memory planning SHALL validate tensor byte-size calculations.

The Runtime SHALL detect:

- element count overflow
- dtype byte-size overflow
- total byte-size overflow
- Provider maximum-size violation
- Device maximum-size violation

#### Scenario: Byte-size overflow

Given a Tensor Descriptor would overflow byte-size calculation

When memory planning occurs

Then the Runtime rejects the plan with a structured size-overflow error.

---

### Requirement: Memory Reuse Safety

The Runtime SHALL preserve correctness when reusing memory for intermediate values.

Memory reuse SHALL preserve correctness, Resource Affinity and observable
resource lifetimes.

#### Scenario: Unsafe reuse

Given an intermediate buffer is still required by a later operation

When the Runtime considers reusing it

Then the Runtime rejects reuse until the buffer is no longer live.

---

### Requirement: Output Resource Ownership

Memory planning SHALL define ownership for produced Tensor Resources.

Produced Tensor Resources SHALL carry Resource Affinity metadata.

#### Scenario: Produce output resource

Given a Compute Graph completes successfully

When output Tensor Resources are returned

Then each output records Provider and Device affinity when applicable.

---

### Requirement: Memory Pressure Reporting

The Runtime SHALL support stable memory pressure diagnostics when reporting memory planning failures.

Memory pressure diagnostics MAY include:

- estimated required bytes
- estimated peak bytes
- selected Provider identifier
- selected Device identifier
- rejected Device memory limits
- materialization cost
- transfer buffer cost

Diagnostics SHALL NOT expose native memory addresses, pointers, allocator
internals or backend storage handles.

#### Scenario: Report memory pressure

Given memory planning fails

When diagnostics are available

Then the Runtime may report stable memory pressure metadata.

---

### Requirement: Provider-Owned Native Allocation

Providers SHALL own native allocation implementation details.

Provider-owned allocation details include:

- allocator implementation
- memory pools
- backend storage representation
- native buffer handles
- queue-specific allocation behavior
- device-specific memory APIs

#### Scenario: Provider executes planned graph

Given the Runtime has validated a Memory Plan

When the Provider executes the graph

Then the Provider may allocate and optimize memory internally without exposing
native allocation details.

---

### Requirement: No Live Resource Migration

The Memory Planning Model SHALL NOT imply live resource migration.

Moving Provider-pinned resources requires explicit transfer, copy,
materialization, replay, reload or a future migration contract.

#### Scenario: Provider-pinned resource unavailable

Given a Provider-pinned Tensor Resource becomes unavailable

When execution depends on it

Then the Runtime reports interruption or failure instead of silently migrating
the resource.

---

### Requirement: Structured Memory Planning Errors

The Runtime SHALL return stable structured errors for memory planning failures.

Structured errors SHALL include categories for:

- memory planning failed
- out of memory
- resource exhausted
- size overflow
- incompatible resource affinity
- unsupported layout
- materialization required
- transfer required
- Provider memory limit exceeded
- Device memory limit exceeded

Backend diagnostics MAY be attached for debugging but SHALL NOT define the
stable contract.

#### Scenario: Report memory planning failure

Given memory planning fails

When the Runtime reports the failure

Then the error uses a stable structured memory planning error variant.

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

### Requirement: Scheduler Model

The Runtime SHALL define a Scheduler Model.

The Scheduler SHALL accept validated Compute Execution Plans.

The Scheduler SHALL NOT accept invalid, unresolved or partially planned compute
work.

#### Scenario: Schedule execution plan

Given a validated Compute Execution Plan

When the Runtime schedules it

Then the Scheduler creates a Scheduled Operation.

---

### Requirement: Runtime-Owned Scheduler

The Scheduler SHALL be owned by the Runtime.

Components SHALL NOT access native Provider queues, Device streams, threads,
locks or execution handles.

#### Scenario: Component submits work

Given a Component submits compute work

When the work is scheduled

Then the Component receives a stable Scheduled Operation resource or result,
not a native execution handle.

---

### Requirement: Scheduled Operation

The Scheduler SHALL represent scheduled work as a Scheduled Operation.

A Scheduled Operation SHALL have a stable identifier.

A Scheduled Operation SHALL expose portable execution state.

#### Scenario: Observe scheduled operation

Given a Scheduled Operation

When its state is queried

Then the Runtime returns a stable scheduling state.

---

### Requirement: Scheduling States

Scheduled Operations SHALL use stable lifecycle states.

The lifecycle states SHALL include:

- accepted
- queued
- ready
- submitted
- running
- completed
- cancelled
- failed
- interrupted

#### Scenario: Operation completes

Given a Scheduled Operation is running

When Provider execution completes successfully

Then the operation reaches the completed terminal state.

---

### Requirement: Terminal States

Completed, cancelled, failed and interrupted SHALL be terminal states.

A Scheduled Operation SHALL NOT leave a terminal state.

#### Scenario: Terminal operation

Given a Scheduled Operation is completed

When it is queried again

Then it remains completed.

---

### Requirement: Execution Plan Preservation

The Scheduler SHALL preserve the Compute Execution Plan constraints.

The Scheduler SHALL NOT change selected Provider, selected Device, Resource
Affinity, Memory Plan or Data Movement steps without creating a new Execution
Plan.

#### Scenario: Preserve selected Provider

Given an Execution Plan selects a Provider

When the Scheduler submits the operation

Then it submits to that Provider unless the operation is rejected before
submission.

---

### Requirement: Provider and Device Availability Check

The Scheduler SHALL check Provider and Device availability before submission
when availability information is available.

#### Scenario: Provider unavailable before submission

Given a Scheduled Operation is queued

And the selected Provider becomes unavailable before submission

When the Scheduler prepares submission

Then the Scheduler fails or interrupts the operation with a structured error
unless a future replanning contract explicitly allows a new plan.

---

### Requirement: No Implicit Replanning

The Scheduler SHALL NOT silently re-resolve or replan Provider-pinned work.

Replanning requires an explicit future contract.

#### Scenario: Provider-pinned work cannot run

Given scheduled work depends on Provider-pinned resources

And the selected Provider is unavailable

When the Scheduler attempts to run it

Then the Scheduler reports failure or interruption instead of selecting another
Provider.

---

### Requirement: Queue Ordering

The Scheduler SHALL define a deterministic queue ordering policy.

The initial Scheduler MAY use FIFO ordering.

Future Scheduling Policies MAY add priority, deadlines, batching or fairness.

#### Scenario: FIFO scheduling

Given two Scheduled Operations are queued with the same priority

When the Scheduler selects the next operation

Then the operation accepted first is selected first.

---

### Requirement: Scheduling Policy

The Scheduler SHALL apply a Scheduling Policy when selecting queued work.

Scheduling Policy SHALL operate on already planned work.

Scheduling Policy SHALL NOT replace Resolution Policy.

#### Scenario: Priority scheduling

Given multiple Scheduled Operations are queued

When a priority Scheduling Policy is active

Then the Scheduler selects work according to scheduling priority without
changing Provider resolution decisions.

---

### Requirement: Backpressure

The Scheduler SHALL define backpressure behavior.

When the Scheduler cannot accept more work, it SHALL reject admission with a
structured backpressure or queue-capacity error.

#### Scenario: Queue full

Given the Scheduler queue is full

When a new Execution Plan is submitted

Then the Scheduler rejects it with a structured queue-capacity error.

---

### Requirement: Cancellation Before Submission

The Scheduler SHALL support cancellation before Provider submission.

#### Scenario: Cancel queued operation

Given a Scheduled Operation is queued

When cancellation is requested

Then the operation reaches the cancelled terminal state without invoking the
Provider.

---

### Requirement: Cancellation After Submission

The Scheduler SHALL support cancellation after Provider submission when the
selected Provider can safely cancel the underlying work.

If cancellation cannot be guaranteed, the Scheduler SHALL report the final
terminal state when Provider execution finishes or fails.

#### Scenario: Cancel running operation

Given a Scheduled Operation is running

When cancellation is requested

Then the Scheduler forwards cancellation to the Provider when supported

And the operation eventually reaches completed, cancelled, failed or interrupted.

---

### Requirement: Completion Observation

The Scheduler SHALL expose completion observation.

Completion observation SHALL return stable terminal state and structured result
or error information.

#### Scenario: Await scheduled operation

Given a Scheduled Operation is running

When the caller awaits completion

Then the Runtime returns completed, cancelled, failed or interrupted terminal
state.

---

### Requirement: Interruption

The Scheduler SHALL distinguish interruption from cancellation and execution
failure.

Interruption means execution cannot continue because of Runtime, Provider,
Device or resource availability failure.

#### Scenario: Device interruption

Given a Scheduled Operation is running

And the selected Device becomes unavailable

When execution cannot continue

Then the Scheduler reports an interrupted terminal state.

---

### Requirement: Provider-Pinned Semantics

The Scheduler SHALL preserve Provider-pinned semantics.

Provider-pinned work SHALL NOT be silently moved to another Provider after state
creation or observable output.

#### Scenario: Provider-pinned session emits output

Given a Provider-pinned operation has emitted observable output

When the Provider fails

Then the Scheduler reports interruption or failure and does not continue on
another Provider.

---

### Requirement: Restartability Awareness

The Scheduler SHALL define restartability awareness for Scheduled Operations.

The Scheduler MAY record whether a Scheduled Operation is transparent,
restartable or Provider-pinned.

When restartability classification is recorded, the Scheduler SHALL expose it as
stable diagnostic metadata rather than as a native backend detail.

The Scheduler SHALL NOT automatically restart work unless a future retry or
replanning policy explicitly permits it.

#### Scenario: Restartable operation fails

Given a Scheduled Operation is classified as restartable

When it fails before observable output

Then the Scheduler may report the restartability hint but does not automatically
replay it unless a retry policy exists.

---

### Requirement: Operation Result

A completed Scheduled Operation SHALL return stable result metadata.

Result metadata MAY include produced Tensor Resources, output descriptors,
usage information, timing diagnostics and execution diagnostics.

Result metadata SHALL NOT expose native Provider handles, backend storage,
queues, streams, GPU pointers or raw memory.

#### Scenario: Completed compute graph

Given a Scheduled Operation completes successfully

When the result is returned

Then produced Tensor Resources include portable descriptors and Resource
Affinity metadata.

---

### Requirement: Scheduler Diagnostics

The Scheduler SHALL define scheduling diagnostics.

The Scheduler MAY produce diagnostics.

When diagnostics are produced, the Scheduler SHALL keep them stable and portable.

Diagnostics MAY include:

- Scheduled Operation identifier
- selected Provider identifier
- selected Device identifier
- queue time
- execution time
- cancellation request time
- terminal state
- stable failure reason

Diagnostics SHALL NOT expose:

- raw backend handles
- queues
- streams
- thread handles
- locks
- GPU pointers
- backend storage
- credentials
- ambient filesystem paths

#### Scenario: Inspect scheduling diagnostics

Given scheduling diagnostics are requested

When the Runtime returns them

Then only stable identifiers, timings and structured reasons are exposed.

---

### Requirement: Structured Scheduler Errors

The Scheduler SHALL return stable structured errors for scheduling failures.

Structured scheduler errors SHALL include categories for:

- invalid execution plan
- queue capacity exceeded
- Provider unavailable
- Device unavailable
- Resource Affinity conflict
- Memory Plan invalid
- submission failed
- cancellation unsupported
- cancellation failed
- execution failed
- execution interrupted
- operation timeout

Backend diagnostics MAY be attached for debugging but SHALL NOT define the
stable contract.

#### Scenario: Report scheduling failure

Given scheduling fails

When the Runtime reports the failure

Then the error uses a stable structured Scheduler error variant.

