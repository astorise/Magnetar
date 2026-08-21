## Purpose

Define the requirements for runtime Providers and capability-based resolution.
## Requirements
### Requirement: Providers

The runtime SHALL use Providers as the native extension mechanism.

Providers expose one or more capabilities to the runtime.

Providers SHALL remain independent from Components.

#### Scenario: Register Provider

Given a valid Provider

When the runtime initializes

Then the Provider is registered.

---

### Requirement: Capability Registration

Every Provider SHALL advertise the capabilities it implements.

#### Scenario: Provider startup

Given a Provider exposing multiple capabilities

When the Provider starts

Then every capability becomes available through the Capability Registry.

---

### Requirement: Capability Resolution

The runtime SHALL resolve Providers through requested capabilities.

Components SHALL never directly reference a Provider.

#### Scenario: Resolve Provider

Given multiple Providers implementing the same capability

When a Component requests that capability

Then the runtime selects a compatible Provider.

---

### Requirement: Provider Fallback

The runtime SHALL support fallback Providers while resolving work that has not
created or consumed Provider-owned state.

Once live resource affinity binds a call to a Provider or Device, the runtime
SHALL only use a Provider that satisfies the complete affinity constraint set.
An unavailable bound Provider SHALL produce a structured affinity failure
instead of implicit migration.

#### Scenario: Primary Provider unavailable before state creation

- **GIVEN** the preferred Provider cannot execute a Capability
- **AND** no live resource affinity constrains the call
- **AND** another compatible Provider exists
- **WHEN** execution is resolved
- **THEN** the runtime selects the fallback Provider

#### Scenario: Bound Provider unavailable after state creation

- **GIVEN** a live opaque resource bound to a Provider
- **WHEN** that Provider becomes unavailable for a dependent call
- **THEN** the runtime reports an affinity failure
- **AND** it does not select another Provider for that live resource

### Requirement: Provider Isolation

Provider failures SHALL remain isolated.

#### Scenario: Provider initialization failure

Given one Provider fails during initialization

When another compatible Provider exists

Then runtime initialization continues.

---

### Requirement: Capability Versioning

Capabilities SHALL be versioned independently from Providers.

For this change, a compatible capability has the same package-qualified name
and exact WIT contract version as the requested capability. Range negotiation
is out of scope until the scheduler is introduced.

#### Scenario: Multiple compatible versions

Given multiple Providers implementing a requested version of a capability

When a Component requests that version

Then the runtime selects a compatible implementation.

---

### Requirement: Component Independence

Components SHALL depend exclusively on WIT capability contracts.

Components SHALL remain independent from native implementations.

#### Scenario: Execute Component

Given the same Component

And different Providers implementing the required capability

When execution occurs

Then the Component executes without modification.

### Requirement: Provider Compute Advertisement

Providers implementing `magnetar:compute/run` SHALL expose a Provider Compute
Advertisement.

The advertisement SHALL describe the Provider's portable compute support.

The advertisement SHALL NOT expose native handles, backend storage, kernel
symbols, queues, streams, locks or raw Device APIs.

#### Scenario: Register compute Provider

Given a Provider implements `magnetar:compute/run`

When the Provider is registered

Then the Runtime records its Provider Compute Advertisement.

---

### Requirement: Capability Version Support

A Provider Compute Advertisement SHALL declare the supported
`magnetar:compute/run` Capability versions.

#### Scenario: Resolve Capability version

Given a Component requires a specific `magnetar:compute/run` version

When the Runtime evaluates Providers

Then only Providers advertising a compatible version are considered.

---

### Requirement: Operation Family Support

A Provider Compute Advertisement SHALL declare supported Compute Operation
Families.

Operation Family support SHALL be used as a coarse compatibility signal.

#### Scenario: Evaluate operation family

Given a Compute Graph contains a linear algebra operation

When the Runtime evaluates a Provider

Then the Provider must advertise compatible linear algebra support.

---

### Requirement: Operation Schema Support

A Provider Compute Advertisement SHALL declare supported Compute Operation
Schemas.

Operation Schema support SHALL be more precise than Operation Family support.

#### Scenario: Unsupported operation schema

Given a Provider supports the linear algebra family

But does not support the requested matrix multiplication schema

When the Runtime validates Provider compatibility

Then the Runtime rejects that Provider for the graph.

---

### Requirement: Portable and Provider-Specific Operations

A Provider Compute Advertisement SHALL distinguish portable operation schemas
from Provider-specific extensions.

Provider-specific extensions SHALL NOT be required by portable Components.

#### Scenario: Provider-specific extension

Given a Compute Graph uses a Provider-specific operation

When the graph is validated as portable compute

Then validation fails unless the operation is explicitly marked as a
Provider-specific extension and the selected Provider advertises support.

---

### Requirement: DType Support

A Provider Compute Advertisement SHALL declare supported dtypes.

DType support MAY vary by operation schema, input position, output position and
Device.

#### Scenario: Unsupported dtype for operation

Given a Provider supports `tensor.matmul`

But only for `f16` and `f32`

When a Compute Graph requests `tensor.matmul` with `i8`

Then the Runtime rejects that Provider for the graph.

---

### Requirement: Layout Support

A Provider Compute Advertisement SHALL declare supported layout constraints.

Layout support MAY include:

- contiguous layout
- portable strided layout
- Provider-managed opaque layout
- view consumption support
- materialization requirement

#### Scenario: View requires materialization

Given a Tensor Resource is a view

And the selected Provider cannot consume that view directly

When the Runtime validates the graph

Then the Runtime requires explicit materialization or rejects execution.

---

### Requirement: Shape and Size Limits

A Provider Compute Advertisement SHALL declare shape and size limits.

Limits MAY include:

- maximum rank
- maximum dimension value
- maximum element count
- maximum byte size
- supported batch dimensions
- broadcasting constraints

#### Scenario: Shape exceeds Provider limit

Given a Compute Graph contains a Tensor Descriptor exceeding a Provider limit

When the Runtime validates Provider compatibility

Then the Runtime rejects that Provider before execution.

---

### Requirement: Precision Support

A Provider Compute Advertisement SHALL support declaring precision constraints.

Precision constraints MAY include:

- accumulation dtype
- approximate math support
- exact math support
- reduced precision support
- mixed precision support

#### Scenario: Precision policy required

Given a Compute Graph requires deterministic exact accumulation

When the Runtime evaluates a Provider

Then the Provider must advertise compatible precision support.

---

### Requirement: Determinism Support

A Provider Compute Advertisement SHALL support declaring deterministic behavior.

Determinism support SHALL be explicit.

The Runtime SHALL NOT assume bitwise equivalent results across Providers.

#### Scenario: Deterministic random generation

Given a Compute Graph requests deterministic random generation

When the Runtime evaluates Providers

Then only Providers advertising compatible deterministic random behavior are
eligible.

---

### Requirement: Data Movement Support

A Provider Compute Advertisement SHALL declare supported data movement paths
when the Provider participates in data movement.

Data movement support MAY include:

- upload
- download
- copy
- transfer
- materialize
- dtype conversion
- layout conversion
- host-staged transfer

#### Scenario: Transfer requires host staging

Given a transfer between two Providers requires host staging

When the Runtime evaluates the movement path

Then the Provider advertisement must indicate whether host staging is required
or unsupported.

---

### Requirement: Device-Specific Advertisement

A Provider Compute Advertisement SHALL support Device-specific compute support
when a Provider exposes different compute support for different Devices.

When support differs by Device, the Provider Compute Advertisement SHALL attach
constraints to stable Device identifiers.

#### Scenario: Multi-device Provider

Given one Provider exposes multiple Devices

And each Device has different memory or dtype support

When the Runtime evaluates candidates

Then Device-specific advertisement data is used during selection.

---

### Requirement: Advertisement and Resource Affinity

The Runtime SHALL evaluate Provider Compute Advertisements together with
Resource Affinity.

A Provider advertisement SHALL NOT override an existing Provider-pinned resource
affinity.

#### Scenario: Provider-pinned tensor

Given a Tensor Resource is bound to one Provider

And another Provider advertises support for the requested operation

When the Tensor Resource is used without explicit transfer

Then the Runtime rejects the second Provider for that operation.

---

### Requirement: Advertisement and Resolution Policy

Resolution Policies SHALL consider Provider Compute Advertisements.

A Provider that implements the requested Capability MAY still be rejected when
its advertisement does not satisfy the graph requirements.

#### Scenario: Capability compatible but graph incompatible

Given a Provider implements `magnetar:compute/run`

But does not advertise support for a required operation schema

When the Runtime resolves the graph

Then the Resolution Policy excludes that Provider.

---

### Requirement: Advertisement Validation

The Runtime SHALL validate Provider Compute Advertisements during Provider
registration.

Invalid advertisements SHALL prevent the affected support entry from being used.

#### Scenario: Invalid advertisement

Given a Provider advertises an unknown operation schema

When the Provider is registered

Then the Runtime rejects or ignores that advertisement entry with diagnostics.

---

### Requirement: Stable Advertisement Values

Provider Compute Advertisements SHALL use stable portable values.

Advertisement values SHALL NOT contain:

- Rust trait objects
- function pointers
- callbacks
- raw native handles
- backend object references
- platform-dependent integer assumptions

#### Scenario: Inspect advertisement

Given a Component or diagnostic tool inspects Provider support metadata

When advertisement data is returned

Then it contains only stable identifiers, versions, limits and portable values.

---

### Requirement: Structured Advertisement Errors

The Runtime SHALL return stable structured errors for advertisement-related
failures.

Structured errors SHALL include categories for:

- invalid advertisement
- unsupported operation schema
- unsupported operation family
- unsupported dtype
- unsupported layout
- unsupported precision policy
- unsupported deterministic behavior
- unsupported data movement
- Device constraint mismatch
- Resource Affinity conflict

Backend diagnostics MAY be attached but SHALL NOT define the stable contract.

#### Scenario: Report advertisement mismatch

Given no Provider advertisement satisfies a Compute Graph

When the Runtime reports the failure

Then it returns a stable structured advertisement or unsupported-feature error.

---

### Requirement: No Execution Guarantee

A Provider Compute Advertisement SHALL describe declared support.

It SHALL NOT guarantee successful execution.

Execution may still fail because of runtime conditions such as memory pressure,
Device unavailability, Provider interruption or resource exhaustion.

#### Scenario: Advertised operation fails at runtime

Given a Provider advertises support for an operation schema

When execution fails due to resource exhaustion

Then the Runtime reports a structured execution error rather than treating the
advertisement as false.

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

