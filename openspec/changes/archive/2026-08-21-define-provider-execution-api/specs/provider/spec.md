## ADDED Requirements

### Requirement: Provider Execution API

The Runtime SHALL define a Provider Execution API.

The Provider Execution API SHALL be the native Runtime-to-Provider interface for
executing validated work.

The Provider Execution API SHALL NOT be exposed as a WIT Capability to portable
Components.

#### Scenario: Submit planned work

Given a Compute Execution Plan has been validated and scheduled

When the Scheduler submits it for execution

Then the Runtime invokes the selected Provider through the Provider Execution
API.

---

### Requirement: Validated Plans Only

Providers SHALL receive only validated Compute Execution Plans.

The Runtime SHALL NOT submit unresolved, partially planned or invalid work to a
Provider.

#### Scenario: Invalid execution plan

Given a Compute Execution Plan has unresolved dependencies

When Provider submission is attempted

Then the Runtime rejects the submission before invoking the Provider.

---

### Requirement: Provider Binding Preservation

The Provider Execution API SHALL preserve the selected Provider from the Compute
Execution Plan.

A Provider SHALL NOT re-resolve execution to another Provider.

#### Scenario: Preserve Provider selection

Given an Execution Plan selects a CUDA Provider

When the Scheduler submits the operation

Then the Runtime submits the operation to that selected Provider only.

---

### Requirement: Device Binding Preservation

The Provider Execution API SHALL preserve the selected Device when the Execution
Plan is Device-bound.

A Provider SHALL NOT silently execute on a different Device when the plan
requires a specific Device.

#### Scenario: Preserve Device selection

Given an Execution Plan selects Device `gpu:0`

When the Provider executes the plan

Then execution occurs on the selected Device or fails with a structured Device
availability or compatibility error.

---

### Requirement: Resource Affinity Preservation

The Provider Execution API SHALL preserve Resource Affinity constraints.

Provider-pinned and Device-bound resources SHALL NOT be silently moved,
materialized, copied or transferred by the Provider unless the Execution Plan
explicitly includes that step.

#### Scenario: Provider-pinned tensor input

Given a Tensor Resource is Provider-pinned

When the Provider receives the Execution Plan

Then the Provider consumes the resource according to its affinity or returns a
structured affinity error.

---

### Requirement: Memory Plan Preservation

The Provider Execution API SHALL preserve the Memory Plan from the Execution
Plan.

Providers MAY optimize native allocation internally.

Providers SHALL NOT violate observable Memory Plan constraints.

#### Scenario: Execute with memory plan

Given an Execution Plan includes a Memory Plan

When the Provider executes the plan

Then temporary buffers, outputs and materialization requirements respect the
planned constraints.

---

### Requirement: Data Movement Preservation

The Provider Execution API SHALL preserve explicit Data Movement steps.

Providers SHALL NOT hide upload, download, copy, transfer, materialization or
host staging when those operations affect observable placement, cost or
synchronization behavior.

#### Scenario: Transfer required

Given an Execution Plan includes an explicit Transfer step

When the Provider executes the plan

Then the Transfer is executed or the operation fails with a structured transfer
error.

---

### Requirement: Provider Execution Handle

Provider submission SHALL return a Provider Execution Handle or a structured
submission error.

The Provider Execution Handle SHALL be Runtime-native.

Components SHALL NOT receive or inspect native Provider Execution Handles.

#### Scenario: Provider submission succeeds

Given a Scheduled Operation is submitted to a Provider

When the Provider accepts the work

Then the Runtime records a Provider Execution Handle internally.

---

### Requirement: Execution Status

The Provider Execution API SHALL allow the Runtime to observe execution status.

Provider status SHALL be mapped to stable Scheduled Operation states.

#### Scenario: Observe running work

Given Provider execution is active

When the Scheduler queries execution status

Then the Runtime maps the Provider status to a stable scheduling state.

---

### Requirement: Completion Result

The Provider Execution API SHALL return a completion result when execution
finishes successfully.

Completion results MAY include produced opaque Tensor Resources and portable
metadata.

Produced Tensor Resources SHALL carry Resource Affinity metadata.

#### Scenario: Return tensor outputs

Given Provider execution completes successfully

When the Runtime collects the result

Then output Tensor Resources include descriptors and Resource Affinity metadata.

---

### Requirement: Cancellation Request

The Provider Execution API SHALL support cancellation requests.

A Provider MAY report that cancellation is unsupported or cannot be guaranteed.

Cancellation SHALL eventually resolve to a terminal Scheduled Operation state.

#### Scenario: Cancel running Provider work

Given a Scheduled Operation is running inside a Provider

When cancellation is requested

Then the Runtime forwards cancellation to the Provider

And the Scheduled Operation eventually reaches completed, cancelled, failed or
interrupted.

---

### Requirement: Cancellation Race Handling

The Provider Execution API SHALL handle cancellation races.

If execution completes before cancellation is applied, completion SHALL remain a
valid terminal state.

#### Scenario: Cancel after completion

Given Provider execution has already completed

When cancellation is requested

Then the Runtime preserves the completed terminal state.

---

### Requirement: Interruption Reporting

The Provider Execution API SHALL report interruptions distinctly from
cancellation and ordinary execution failure.

Interruption means execution cannot continue because of Provider, Device,
Runtime or resource availability failure.

#### Scenario: Provider interruption

Given Provider execution is running

When the Provider becomes unavailable

Then the Runtime reports an interrupted terminal state.

---

### Requirement: Stable Error Mapping

Provider-native errors SHALL be mapped to stable Runtime error categories.

Backend-specific diagnostics MAY be attached.

Backend diagnostics SHALL NOT define the stable contract.

#### Scenario: Native backend error

Given a Provider returns a CUDA, Metal, CPU, OpenVINO or other backend-specific
error

When the Runtime reports it

Then the Runtime maps it to a stable Magnetar error category and attaches native
details only as diagnostics.

---

### Requirement: Native Detail Privacy

The Provider Execution API SHALL NOT expose native execution details to
Components.

Forbidden exposed values include:

- raw pointers
- GPU pointers
- backend storage
- queues
- streams
- threads
- locks
- file descriptors
- allocator internals
- kernel symbols
- Provider handles
- Device handles

#### Scenario: Inspect scheduled operation

Given a Component observes a Scheduled Operation

When the Runtime returns execution metadata

Then it returns stable identifiers and portable metadata only.

---

### Requirement: Provider-Owned Native Execution

Providers SHALL own native execution implementation details.

Native execution details include:

- kernel selection
- kernel fusion
- memory allocation
- backend storage
- queue submission
- stream synchronization
- hardware-specific optimization
- device-specific APIs

#### Scenario: Execute native work

Given a Provider receives a valid Execution Plan

When it executes the work

Then it may use native mechanisms internally without exposing them through the
portable Runtime contract.

---

### Requirement: No Provider-Side Replanning

Providers SHALL NOT change Runtime planning decisions.

A Provider MAY reject execution if the plan is no longer valid or executable.

A Provider SHALL NOT silently choose another Provider, another incompatible
Device, or unplanned data movement.

#### Scenario: Plan no longer executable

Given a Provider receives an Execution Plan

And the selected Device can no longer satisfy the plan

When execution is attempted

Then the Provider returns a structured execution or Device availability error.

---

### Requirement: No Automatic Migration

The Provider Execution API SHALL NOT imply automatic migration of live state.

Moving Provider-pinned resources requires explicit transfer, copy,
materialization, replay, reload or a future migration contract.

#### Scenario: Provider-pinned resource fails

Given work depends on Provider-pinned live resources

When the Provider fails during execution

Then the Runtime reports interruption or failure instead of silently migrating
the work.

---

### Requirement: Execution Resource Release

The Provider Execution API SHALL define release behavior for Provider-owned
temporary execution resources.

Temporary execution resources SHALL be released after terminal state unless
retained by an output resource or explicit Runtime-owned resource.

#### Scenario: Release temporary resources

Given Provider execution reaches a terminal state

When the Runtime collects the result

Then temporary Provider execution resources are released according to Provider
lifecycle rules.

---

### Requirement: Execution Diagnostics

The Provider Execution API SHALL support optional diagnostics.

The Provider Execution API MAY return diagnostics.

Diagnostics MAY include:

- Provider identifier
- Device identifier
- execution phase
- stable failure reason
- timing information
- memory pressure metadata
- backend diagnostic string
- trace identifier

Diagnostics SHALL NOT expose native handles, raw pointers, credentials,
filesystem secrets or backend-private object references.

#### Scenario: Report Provider diagnostic

Given Provider execution fails

When diagnostics are available

Then the Runtime records stable diagnostic metadata and redacted backend details.

---

### Requirement: Structured Provider Execution Errors

The Runtime SHALL return stable structured errors for Provider execution API
failures.

Structured errors SHALL include categories for:

- Provider unavailable
- Device unavailable
- invalid execution plan
- incompatible resource affinity
- memory plan rejected
- unsupported operation
- unsupported dtype
- unsupported layout
- data movement failed
- materialization failed
- submission failed
- execution failed
- execution interrupted
- cancellation unsupported
- cancellation failed
- resource exhausted
- out of memory

Backend diagnostics MAY be attached for debugging but SHALL NOT define the
stable contract.

#### Scenario: Report Provider execution failure

Given Provider execution fails

When the Runtime reports the failure

Then the error uses a stable structured Provider execution error category.
