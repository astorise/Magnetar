## ADDED Requirements

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
