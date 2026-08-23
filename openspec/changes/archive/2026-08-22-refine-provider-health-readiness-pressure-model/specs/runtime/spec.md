## ADDED Requirements

### Requirement: Runtime Records Provider Status Snapshots

Runtime SHALL record Provider status as immutable snapshots for decision points.

#### Scenario: Status changes during resolution

Given a Provider status changes while Resolution is running

When Resolution starts with a recorded snapshot

Then the decision is based on the captured snapshot

And later changes are handled by subsequent checks or retries.

---

### Requirement: Runtime Evaluates Status Freshness

Runtime SHALL evaluate Provider status freshness before using it for Resolution
or Scheduling.

#### Scenario: Expired report

Given a Provider report TTL has expired

When Runtime evaluates new work

Then Runtime treats the status as stale according to policy.

---

### Requirement: Runtime Distinguishes Readiness From Health

Runtime SHALL not treat a healthy Provider as ready unless readiness also allows
admission.

#### Scenario: Warmup

Given a Provider reports healthy

And readiness reports not-ready

When Runtime resolves new work

Then the Provider is not selected for ordinary new work.

---

### Requirement: Runtime Distinguishes Pressure From Failure

Runtime SHALL not classify pressure alone as Provider failure.

#### Scenario: Saturation

Given a Provider reports saturated pressure

When Runtime rejects new work due to admission policy

Then the Provider is not automatically marked failed.

---

### Requirement: Runtime Handles Provider Drainage

Runtime SHALL support Provider drainage behavior.

During drain, Runtime SHALL prevent ordinary new unpinned work while preserving
safe handling of in-flight and pinned work according to policy.

#### Scenario: Drain requested

Given Runtime marks a Provider as draining

When new unpinned work arrives

Then Runtime avoids selecting the draining Provider.

---

### Requirement: Runtime Does Not Silently Migrate During Drain

Runtime SHALL not silently move Provider-owned resources merely because a
Provider is draining.

#### Scenario: Provider-bound tensor

Given a tensor is bound to a draining Provider

When a dependent operation is submitted

Then Runtime either continues according to pinned-resource policy or requires
explicit data movement.

---

### Requirement: Runtime Checks Status Before Submission

After planning and before Provider submission, Runtime or Scheduler SHALL verify
that selected Provider and Device status still permits admission.

#### Scenario: Provider becomes saturated

Given Resolution selected Provider A

And Provider A becomes saturated before submission

When Scheduler attempts admission

Then Runtime policy decides whether to retry resolution, queue, or fail.

---

### Requirement: Runtime Normalizes Provider Refusal

Provider refusal due to readiness, pressure, draining, or stale status SHALL be
normalized into stable Runtime errors.

#### Scenario: Provider refuses admission

Given Provider rejects work because readiness is not-ready

When Runtime receives the refusal

Then Runtime reports a Provider-not-ready style error

And not an execution-failed error.

---

### Requirement: Runtime Emits Provider Status Observability

Runtime SHALL support structured observations for Provider lifecycle, health,
readiness, pressure, admission, staleness, drainage, and recovery events.

Observability SHALL not become the source of truth for status.

#### Scenario: Provider pressure rises

Given Provider pressure changes from moderate to high

When Runtime observes the change

Then an observation may be emitted

And scheduling behavior remains based on Runtime status state and policy.

---

### Requirement: Runtime Policy Controls Degraded And Pressured Providers

Runtime policy SHALL define how degraded, pressured, saturated, draining, and
stale Providers are handled.

#### Scenario: Development policy allows stale Provider

Given development policy allows stale Provider status

When a Provider report expires

Then Runtime may continue using the Provider according to that policy

But the decision is explicit.
