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

### Requirement: Provider Is Sole Native Execution Extension

Provider SHALL be the only active trusted native execution extension mechanism.

Provider implementations SHALL own native hardware execution details and expose
Devices through Provider metadata and registration.

Provider implementations SHALL NOT depend on a separate Backend or Plugin
registry to participate in Runtime resolution.

#### Scenario: Register native implementation

Given a native implementation exposes execution capabilities

When it is registered with the Runtime

Then it is registered as a Provider

And its Devices and Capability advertisements are evaluated through Provider
resolution.

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

### Requirement: Provider Health Model

The Runtime SHALL define a Provider Health Model.

The Provider Health Model SHALL describe Provider and Device availability using
stable portable states.

The Provider Health Model SHALL NOT expose native handles, driver objects,
queues, streams, backend storage or Provider-private internals.

#### Scenario: Report Provider health

Given a Provider is registered

When the Runtime queries Provider health

Then the Runtime receives a stable Provider Health Report.

---

### Requirement: Health States

The Provider Health Model SHALL define stable health states.

Health states SHALL include:

- unknown
- initializing
- available
- degraded
- saturated
- draining
- unavailable
- interrupted

#### Scenario: Provider available

Given a Provider is ready to accept work

When it reports health

Then its health state is `available`.

---

### Requirement: Unknown Health

A Provider or Device with unknown health SHALL NOT be treated as healthy unless
the active policy explicitly allows unknown health.

#### Scenario: Unknown Provider health

Given a Provider has no fresh health report

When the Runtime evaluates it for scheduling

Then the Runtime treats health as `unknown`.

---

### Requirement: Initializing Health

The Runtime SHALL support Providers reporting `initializing` while they are starting or discovering
Devices.

The Runtime SHALL treat `initializing` as not ready for work by default.

Providers in `initializing` state SHALL NOT receive execution work unless the
active policy explicitly permits it.

#### Scenario: Provider initializing

Given a Provider is still discovering Devices

When the Scheduler evaluates it

Then the Scheduler does not submit work to it by default.

---

### Requirement: Available Health

The Runtime SHALL allow a Provider or Device in `available` state to be considered for resolution,
planning, scheduling and execution.

The Runtime SHALL preserve structured execution errors even when prior health
was `available`.

Availability SHALL NOT guarantee successful execution.

#### Scenario: Available Provider fails

Given a Provider reports `available`

When execution later fails due to runtime conditions

Then the Runtime reports a structured execution error without treating the
previous health report as a contract violation.

---

### Requirement: Degraded Health

The Runtime SHALL support Providers and Devices reporting `degraded` when they can accept work but with
reduced reliability, capacity or performance.

The Runtime SHALL expose degraded health to Resolution Policies and Scheduling
Policies.

Resolution Policies and Scheduling Policies MAY reject or deprioritize degraded
Providers.

#### Scenario: Degraded Provider

Given a Provider reports `degraded`

When the Resolution Policy prefers healthy Providers

Then the degraded Provider is ranked lower or rejected.

---

### Requirement: Saturated Health

The Runtime SHALL support Providers and Devices reporting `saturated` when they cannot currently accept more
work because of capacity, memory pressure, queue pressure or execution limits.

The Runtime SHALL expose saturation as a stable health state.

The Scheduler MAY treat saturation as backpressure.

#### Scenario: Saturated Provider

Given a Provider reports `saturated`

When the Scheduler attempts to admit new work

Then the Scheduler delays or rejects admission with a structured backpressure or
Provider saturated error.

---

### Requirement: Draining Health

The Runtime SHALL support Providers reporting `draining` when they should finish existing work but should
not receive new work.

The Scheduler SHALL avoid assigning new work to draining Providers by default.

#### Scenario: Provider draining

Given a Provider reports `draining`

When new work is scheduled

Then the Scheduler avoids submitting new work to that Provider.

---

### Requirement: Unavailable Health

A Provider or Device in `unavailable` state SHALL NOT receive new execution
work.

#### Scenario: Provider unavailable

Given a Provider reports `unavailable`

When the Runtime evaluates it for a new Execution Plan

Then the Runtime rejects that Provider candidate.

---

### Requirement: Interrupted Health

The Runtime SHALL support Providers and Devices reporting `interrupted` when running work can no longer
continue because of Provider, Device, resource or Runtime failure.

The Runtime SHALL map interrupted health to a terminal interrupted or failed
operation outcome when running work cannot continue.

Interrupted running work SHALL reach a failed or interrupted terminal state.

#### Scenario: Running work interrupted

Given a Scheduled Operation is running

And the selected Provider reports `interrupted`

When execution cannot continue

Then the Scheduled Operation reaches an interrupted or failed terminal state.

---

### Requirement: Provider Health Report

A Provider Health Report SHALL include stable Provider identity and health
state.

A Provider Health Report MAY include:

- Provider identifier
- Provider health state
- timestamp
- time-to-live
- diagnostic code
- capacity hints
- supported Device health summaries

#### Scenario: Inspect Provider health report

Given a Provider Health Report

When diagnostics are requested

Then the Runtime returns stable identifiers, health state and redacted
diagnostics only.

---

### Requirement: Device Health Report

Providers that expose Devices SHALL report Device Health when available.

Device Health SHALL use stable Device identifiers.

#### Scenario: Device unavailable

Given a Provider exposes multiple Devices

And one Device becomes unavailable

When the Runtime evaluates execution candidates

Then only compatible available Devices remain eligible.

---

### Requirement: Capability Health

Providers SHALL be able to report health per Capability implementation.

The Runtime SHALL use Capability Health when it is available for the requested
Capability.

A Provider may be available while a specific Capability implementation is
unavailable, degraded or saturated.

#### Scenario: Capability unavailable

Given a Provider is available

But its `magnetar:compute/run` implementation is unavailable

When the Runtime resolves compute work

Then the Provider is rejected for that Capability.

---

### Requirement: Health Freshness

Health reports SHALL include freshness metadata when available.

Freshness metadata MAY include a timestamp or time-to-live.

Stale health reports SHALL NOT be treated as definitely healthy.

#### Scenario: Stale health report

Given a Provider health report has expired

When the Scheduler evaluates the Provider

Then the Scheduler treats health as unknown or requests a fresh report.

---

### Requirement: Resolution Policy Integration

Resolution Policies SHALL consider Provider Health when selecting among
compatible Providers.

A Provider that implements a Capability MAY still be rejected because of health.

#### Scenario: Healthy Provider preferred

Given two Providers implement the requested Capability

And one is available while the other is degraded

When the active Resolution Policy prefers healthy Providers

Then the available Provider is selected.

---

### Requirement: Execution Planning Integration

Execution Planning SHALL consider Provider and Device Health before creating a
final Execution Plan when health information is available.

#### Scenario: Device unavailable during planning

Given a candidate Device is unavailable

When Execution Planning evaluates candidates

Then the Runtime rejects that Device for the plan.

---

### Requirement: Scheduler Admission Integration

The Scheduler SHALL check Provider and Device Health before submitting scheduled
work when health information is available.

#### Scenario: Provider becomes unavailable before submission

Given a Scheduled Operation is queued

And the selected Provider becomes unavailable before submission

When the Scheduler prepares to submit work

Then the Scheduler fails or interrupts the operation unless a future replanning
contract explicitly allows a new plan.

---

### Requirement: Provider Execution API Integration

The Provider Execution API SHALL surface Provider and Device health-related
failures as stable Runtime errors.

#### Scenario: Device lost during execution

Given a Provider is executing work

And the selected Device becomes unavailable

When the Provider reports the failure

Then the Runtime maps it to a stable Device unavailable, execution failed or
execution interrupted error.

---

### Requirement: Health Is Not Failover

Provider Health SHALL NOT imply automatic failover.

The Runtime SHALL NOT silently migrate live Provider-pinned resources when
Provider Health changes.

#### Scenario: Provider-pinned work loses Provider

Given running work depends on Provider-pinned resources

And the owning Provider becomes unavailable

When the Runtime observes the health change

Then the Runtime reports interruption or failure instead of silently continuing
on another Provider.

---

### Requirement: No Implicit Replanning

Health changes SHALL NOT cause implicit replanning of Provider-pinned work.

Replanning requires a future explicit contract.

#### Scenario: Alternative Provider available

Given a Provider-pinned operation is running

And another Provider advertises compatible support

When the original Provider becomes unavailable

Then the Runtime does not silently switch to the other Provider.

---

### Requirement: Backpressure

Provider Health SHALL be able to express backpressure through the `saturated` state or
capacity hints.

The Scheduler SHALL expose backpressure as a stable scheduling or admission
outcome.

Backpressure SHALL be reported as a stable scheduling or admission outcome.

#### Scenario: Provider queue saturated

Given a Provider reports queue saturation

When the Scheduler attempts to submit additional work

Then the Scheduler delays or rejects admission with a stable backpressure error.

---

### Requirement: Capacity Hints

Provider Health Reports SHALL be able to include capacity hints.

The Runtime SHALL treat capacity hints as advisory metadata.

Capacity hints MAY include:

- queue depth
- available memory estimate
- memory pressure
- active operation count
- maximum accepted operations
- recommended admission limit

Capacity hints SHALL be advisory.

#### Scenario: Capacity hint used

Given a Provider reports limited available memory

When the Scheduler evaluates new work

Then the Scheduler may reject or delay work according to policy.

---

### Requirement: Stable Health Diagnostics

Provider Health diagnostics SHALL use stable portable values.

Diagnostics MAY include:

- Provider identifier
- Device identifier
- health state
- diagnostic code
- timestamp
- trace identifier
- redacted backend diagnostic message

Diagnostics SHALL NOT expose:

- raw pointers
- GPU pointers
- queues
- streams
- locks
- file descriptors
- kernel symbols
- backend storage
- Provider handles
- Device handles
- credentials
- ambient filesystem paths

#### Scenario: Inspect health diagnostics

Given a Provider Health Report contains diagnostics

When the Runtime exposes diagnostics

Then only stable identifiers and redacted diagnostic values are returned.

---

### Requirement: Health State Transitions

The Runtime SHALL tolerate Provider Health state transitions.

The Runtime SHALL preserve terminal Scheduled Operation states even if Provider
Health changes later.

#### Scenario: Health changes after completion

Given a Scheduled Operation has completed

When the Provider later becomes unavailable

Then the completed operation remains completed.

---

### Requirement: Health and Resource Affinity

Provider Health SHALL be evaluated together with Resource Affinity.

Health changes SHALL NOT override existing Resource Affinity constraints.

#### Scenario: Affinity-bound resource

Given a Tensor Resource is bound to one Provider

And that Provider becomes degraded

When dependent work is planned

Then the Runtime either uses the same Provider according to policy, requires an
explicit transfer/replay path, or rejects the work.

---

### Requirement: Structured Health Errors

The Runtime SHALL return stable structured errors for health-related failures.

Structured health errors SHALL include categories for:

- Provider health unknown
- Provider initializing
- Provider degraded
- Provider saturated
- Provider draining
- Provider unavailable
- Provider interrupted
- Device health unknown
- Device degraded
- Device saturated
- Device unavailable
- stale health report
- Capability implementation unavailable

Backend diagnostics MAY be attached for debugging but SHALL NOT define the
stable contract.

#### Scenario: Report Provider unavailable

Given a Provider is unavailable

When work is scheduled or submitted to that Provider

Then the Runtime reports a stable structured Provider unavailable error.

### Requirement: Provider Lifecycle State

A Provider SHALL expose or be associated with a Runtime-managed lifecycle state.

Lifecycle state SHALL describe Runtime management stage, not complete execution
admission.

Lifecycle states SHOULD include:

- registered
- loading
- initializing
- ready
- draining
- stopped
- failed
- removed

#### Scenario: Provider initializing

Given a Provider is loaded but still warming internal state

When Runtime inspects its lifecycle

Then lifecycle may be `initializing`

And the Provider is not yet treated as ready for ordinary new work.

---

### Requirement: Provider Health State

A Provider SHALL report health independently from readiness.

Health states SHOULD include:

- unknown
- healthy
- degraded
- unhealthy
- failed

#### Scenario: Healthy but not ready

Given a Provider has no internal fault

And it is still warming model-related execution state

When health and readiness are reported

Then health may be `healthy`

And readiness may be `not-ready`.

---

### Requirement: Provider Readiness State

A Provider SHALL report whether it should receive new work.

Readiness states SHOULD include:

- not-ready
- ready
- read-only
- draining

#### Scenario: Draining Provider

Given a Provider is draining

When new unpinned work is resolved

Then the Provider is not selected for ordinary new work.

---

### Requirement: Provider Pressure State

A Provider SHALL report pressure independently from health.

Pressure states SHOULD include:

- unknown
- low
- moderate
- high
- saturated

#### Scenario: Saturated Provider

Given a Provider is internally healthy

And all admission capacity is exhausted

When pressure is reported

Then health may remain healthy

And pressure is saturated.

---

### Requirement: Provider Admission Decision

A Provider status model SHALL support admission decisions.

Admission SHOULD distinguish:

- admit
- prefer-not
- reject

Admission MAY be scoped to Provider, Device, Capability, or operation family.

#### Scenario: Prefer not due to pressure

Given a Provider is under high pressure

When admission is evaluated for new work

Then admission may be `prefer-not`

And Resolution Policy may choose another compatible Provider.

---

### Requirement: Provider Status Snapshot

Provider status SHALL be represented as a snapshot.

A snapshot SHALL include sufficient information for Runtime Resolution and
Scheduling to make stable decisions.

A snapshot SHOULD include lifecycle, health, readiness, pressure, admission,
freshness, Device status, and Capability status.

#### Scenario: Status captured

Given a Provider reports status

When Runtime records it

Then the status is captured as an immutable snapshot for that decision point.

---

### Requirement: Provider Status Freshness

Provider status SHALL include freshness metadata.

Runtime SHALL treat expired status as stale.

#### Scenario: Status TTL expires

Given the last Provider status report exceeded its TTL

When Runtime evaluates the Provider

Then the Provider status is considered stale

And policy decides whether to reject, retry, or degrade confidence.

---

### Requirement: Provider Drainage

A Provider SHALL support a draining state or equivalent Runtime-managed
drainage behavior.

A draining Provider SHALL stop receiving ordinary new unpinned work.

It MAY continue existing work where safe.

#### Scenario: Drain Provider

Given Runtime initiates Provider drain

When the Provider enters draining state

Then new ordinary work is rejected or redirected

And in-flight work may complete.

---

### Requirement: Provider Drain Does Not Migrate Resources

Draining a Provider SHALL NOT silently migrate Provider-owned resources.

Any migration, transfer, copy, or materialization SHALL require explicit Runtime
data movement semantics.

#### Scenario: Pinned tensor during drain

Given a tensor is Provider-pinned to a draining Provider

When dependent work requires that tensor

Then Runtime either admits compatible pinned work according to policy

Or returns a structured affinity error

But does not silently move the tensor.

---

### Requirement: Device-Level Provider Status

A Provider exposing Devices SHALL report or allow Runtime to derive
Device-level status.

Device-level status MAY include health, readiness, pressure, memory pressure,
availability, and interruption state.

#### Scenario: One Device unavailable

Given a Provider exposes two Devices

And one Device is unavailable

When Runtime resolves work

Then it may still use the available Device

And reject the unavailable Device.

---

### Requirement: Capability-Level Provider Status

A Provider implementing multiple Capabilities SHALL report or allow Runtime to
derive Capability-level status.

Capability-level status may affect Capability Resolution.

#### Scenario: Compute ready, generation not ready

Given a Provider supports Compute and Generation

And only Compute is ready

When a Component requests Compute

Then the Provider may be eligible

But a Generation request is rejected or delayed.

---

### Requirement: Operation-Family Status

A Provider operation-family status, when reported, SHALL be considered during
validation, planning, or resolution where applicable.

#### Scenario: Matmul saturated

Given a Provider reports linear algebra operations as saturated

When a Compute graph dominated by matrix multiplication is resolved

Then policy may penalize or reject that Provider.

---

### Requirement: Provider Interruption State

Provider status SHALL be able to represent interruption-related conditions.

Examples include driver loss, device reset, device removal, OOM recovery,
thermal throttling, allocator failure, and administrative drain.

#### Scenario: Device reset

Given a GPU Device reset is reported

When Runtime evaluates Provider status

Then Device health, readiness, and admission reflect the interruption.

---

### Requirement: Provider Refusal Is Not Execution Failure

A Provider refusal because it is not ready, draining, or saturated SHALL be
distinguished from execution failure after submission.

#### Scenario: Provider rejects admission

Given Runtime tries to submit work

And Provider rejects because it is saturated

Then Runtime reports an admission/status failure

And not a kernel execution failure.

---

### Requirement: Provider Status Diagnostics

Provider status SHALL provide stable diagnostic reasons suitable for Runtime
logging, Resolution explanation, and observability.

Diagnostics SHALL be redacted and SHALL NOT expose unsafe native handles.

#### Scenario: Provider skipped

Given a Provider is skipped during Resolution because it is draining

When diagnostics are requested

Then the diagnostic reports draining as the stable reason.

### Requirement: Provider Loading Modes

Magnetar SHALL distinguish Provider loading modes.

Supported modes MAY include:

- built-in
- dynamic-library
- test-provider
- development-provider

All modes SHALL register through the Runtime Provider Registry.

#### Scenario: Built-in Provider

Given a Provider is compiled into the Runtime binary

When Runtime initializes

Then it may register through the same Provider Registry as other Providers

Without defining the dynamic library ABI.

---

### Requirement: Dynamic Provider ABI Is Explicit

Dynamic Provider libraries SHALL use an explicit, versioned native ABI.

The stable dynamic ABI SHALL NOT be an implicit Rust trait-object boundary.

#### Scenario: Load dynamic Provider

Given a native library is discovered

When Runtime loads it

Then Runtime performs ABI negotiation through a stable descriptor

And does not accept a Rust `Box<dyn Provider>` as the stable compatibility
contract.

---

### Requirement: Rust Trait Objects Are In-Process Only

Rust trait objects SHALL remain limited to built-in Providers, mocks, or internal
adapters compiled together with the Runtime.

Rust trait objects SHALL NOT be the stable cross-dynamic-library Provider ABI.

#### Scenario: Mock Provider test

Given a unit test uses an in-process Rust mock implementing `Provider`

When the test runs

Then this does not imply that dynamic libraries may return `dyn Provider` as
their stable ABI.

---

### Requirement: Provider Factory Symbol

A dynamic Provider library SHALL expose a canonical factory or descriptor
symbol.

The symbol SHALL allow the Runtime to obtain ABI version and descriptor
information before registration.

#### Scenario: Missing factory symbol

Given a dynamic library lacks the required Provider factory symbol

When Runtime attempts to load it

Then loading fails before Provider registration.

---

### Requirement: Provider ABI Version

Provider dynamic ABI SHALL have an explicit ABI version.

Runtime SHALL reject unsupported ABI major versions.

#### Scenario: Unsupported ABI major

Given a Provider library reports ABI major version 99

When Runtime supports only ABI major version 1

Then the Provider is rejected.

---

### Requirement: Provider ABI Descriptor

A dynamic Provider SHALL expose a descriptor containing required ABI functions
and metadata accessors.

The descriptor SHALL be validated before registration.

#### Scenario: Missing required function

Given a Provider descriptor lacks a required status function

When Runtime validates the descriptor

Then loading fails before Provider registration.

---

### Requirement: Provider Loading Handshake

Runtime SHALL complete a loading handshake before registering a dynamic
Provider.

The handshake SHALL validate ABI, metadata, Capability advertisements, Device
metadata, status reporting, and execution API availability.

#### Scenario: Invalid advertisement

Given a Provider reports malformed Capability advertisements

When the loading handshake runs

Then the Provider is rejected before it becomes eligible for Resolution.

---

### Requirement: Provider Metadata Before Registration

Provider metadata SHALL be retrieved and validated before Provider
registration.

#### Scenario: Duplicate ProviderId

Given a dynamic Provider reports a ProviderId already registered

When Runtime validates metadata

Then loading fails or follows explicit duplicate policy.

---

### Requirement: Provider Capability Advertisements Through ABI

Dynamic Providers SHALL expose Capability advertisements through the Provider
ABI.

Advertisements SHALL be validated before Provider eligibility.

#### Scenario: Advertise Compute

Given a Provider advertises Compute Capability support

When Runtime validates the advertisement

Then malformed or incompatible advertisements are rejected.

---

### Requirement: Provider Device Metadata Through ABI

Dynamic Providers SHALL expose Device metadata through the Provider ABI.

Device metadata SHALL not expose raw native handles as public Runtime API.

#### Scenario: List GPU Device

Given a Provider exposes a GPU Device

When Runtime reads Device metadata

Then Runtime receives stable Device metadata

And not a raw CUDA, HIP, Metal, or driver handle.

---

### Requirement: Provider Status Through ABI

Dynamic Providers SHALL expose status through the ABI using the refined status
model.

Status SHALL include or support lifecycle, health, readiness, pressure,
admission, freshness, Device status, and Capability status.

#### Scenario: Provider saturated

Given a dynamic Provider reports saturated pressure

When Runtime reads status through the ABI

Then Runtime can distinguish saturation from Provider failure.

---

### Requirement: Provider Execution Through ABI

Dynamic Providers SHALL expose execution behavior through a stable ABI-compatible
execution surface.

The ABI SHALL preserve ProviderExecutionApi semantics without exposing arbitrary
Rust types.

#### Scenario: Submit execution

Given Runtime submits Provider execution through dynamic ABI

When the Provider accepts work

Then request and response payloads follow ABI-compatible structures

And not private Rust layouts.

---

### Requirement: ABI Memory Ownership

All memory crossing the Provider ABI boundary SHALL have explicit ownership.

#### Scenario: Provider returns string

Given Provider returns a diagnostic string

When Runtime consumes it

Then the ABI defines whether Runtime must call a Provider release function

And Runtime does not free the memory with the wrong allocator.

---

### Requirement: Provider Opaque Handles

Dynamic Provider state SHALL use opaque handles only as Runtime-internal native
state handles.

Opaque handles SHALL have explicit destroy or release functions.

Opaque handles SHALL not be exposed through Component WIT or public portable
APIs.

#### Scenario: Provider-owned tensor handle

Given Provider returns an opaque native resource handle

When Runtime stores it internally

Then the handle remains Runtime/Provider internal

And is not exposed to portable Components.

---

### Requirement: No Unwind Across Provider ABI

Provider calls SHALL NOT unwind across the ABI boundary.

#### Scenario: Provider panics

Given a Rust-based Provider panics internally

When execution crosses the ABI boundary

Then the Provider adapter catches or handles the panic according to policy

And Runtime receives a stable failure or marks the Provider failed.

---

### Requirement: Provider ABI Error Model

Provider ABI calls SHALL report stable error categories that Runtime can
normalize.

#### Scenario: Provider not ready

Given Provider rejects work because it is not ready

When the error crosses the ABI

Then Runtime maps it to a Provider-not-ready style error

And not to an opaque native failure.

---

### Requirement: Provider Threading Declaration

A dynamic Provider SHALL declare threading and reentrancy expectations.

Runtime SHALL respect the declaration.

#### Scenario: Single-threaded Provider

Given a Provider declares single-threaded execution

When Runtime schedules calls into it

Then Runtime serializes access or rejects the Provider according to policy.

---

### Requirement: Provider Blocking Behavior Declaration

A dynamic Provider SHALL declare relevant blocking or asynchronous execution
behavior.

#### Scenario: Long-running blocking Provider

Given a Provider declares that execution calls may block

When Runtime uses the Provider

Then Runtime isolates or schedules those calls according to Runtime policy.

---

### Requirement: Provider Library Unloading Safety

Runtime SHALL not unload a dynamic Provider library while Provider code may
still be referenced.

#### Scenario: In-flight operation

Given a Provider has an in-flight operation

When Runtime stops the Provider

Then Runtime does not unload the library until unloading is safe

Or it follows a conservative never-unload policy.

---

### Requirement: Provider Loading Is Trusted Native Code

Dynamic Provider loading SHALL be treated as trusted native code execution.

The ABI boundary is not a sandbox.

#### Scenario: Configure Provider path

Given Runtime is configured to load a Provider library

When policy evaluates the path

Then only allowed paths or trusted Provider packages are loaded.

