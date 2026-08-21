## ADDED Requirements

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
