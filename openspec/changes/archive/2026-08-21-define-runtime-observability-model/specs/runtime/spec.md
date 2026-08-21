## ADDED Requirements

### Requirement: Runtime Observability

The Runtime SHALL expose a portable observability model.

Observability SHALL describe Runtime decisions without exposing Provider-native
implementation details.

#### Scenario: Observe Runtime execution

Given compute work executes

When observability is enabled

Then the Runtime emits structured observations describing execution.

---

### Requirement: Runtime Events

The Runtime SHALL emit structured Runtime Events.

Events SHALL include stable Runtime identifiers.

Events SHALL NOT expose native Provider handles.

#### Scenario: Execution starts

Given a Scheduled Operation starts

When execution begins

Then the Runtime emits an execution-start event.

---

### Requirement: Runtime Traces

The Runtime SHALL support correlated execution traces.

Trace identifiers SHALL remain stable during the lifetime of one execution.

#### Scenario: Trace execution

Given a Compute Execution Plan

When execution proceeds

Then every Runtime Event references the same TraceId.

---

### Requirement: Runtime Metrics

The Runtime SHALL expose structured metrics.

Metrics MAY include:

- queue latency
- planning latency
- execution latency
- transfer count
- transfer bytes
- memory estimate
- execution duration
- Provider utilization

#### Scenario: Collect metrics

Given execution completes

When metrics are collected

Then the Runtime reports portable metric values.

---

### Requirement: Runtime Diagnostics

Diagnostics SHALL reference Runtime identifiers.

Diagnostics SHALL NOT expose Provider-native implementation details.

#### Scenario: Report execution failure

Given Provider execution fails

When diagnostics are emitted

Then diagnostics reference Runtime objects and stable error categories.

---

### Requirement: Correlation

Every Runtime Event SHALL be able to reference:

- TraceId
- SpanId
- CorrelationId
- ScheduledOperationId
- ComputeExecutionPlanId
- ProviderId
- DeviceId

#### Scenario: Correlate events

Given multiple Runtime Events

When an observer reconstructs execution

Then events can be correlated without backend-specific identifiers.

---

### Requirement: Privacy

Observability SHALL redact Provider-native implementation details.

Forbidden values include:

- raw pointers
- backend queues
- streams
- native handles
- kernel addresses
- allocator internals
- backend storage objects
- credentials

#### Scenario: Export trace

Given Runtime observations are exported

When an external tool consumes them

Then no Provider-private implementation detail is exposed.

---

### Requirement: Stable Schema

Runtime observations SHALL use stable portable schemas.

Future Runtime implementations SHALL preserve compatibility for existing
observation types.

#### Scenario: Upgrade Runtime

Given a newer Runtime version

When existing observability tooling consumes observations

Then existing observation types remain compatible.
