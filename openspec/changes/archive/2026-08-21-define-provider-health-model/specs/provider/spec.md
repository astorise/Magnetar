## ADDED Requirements

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
