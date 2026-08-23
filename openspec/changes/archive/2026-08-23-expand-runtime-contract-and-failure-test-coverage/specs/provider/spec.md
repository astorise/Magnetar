## ADDED Requirements

### Requirement: Provider Status Test Coverage

Provider status dimensions SHALL be covered by tests.

Tests SHALL distinguish lifecycle, health, readiness, pressure, admission, and
freshness.

#### Scenario: Healthy but saturated

Given a Provider reports healthy health

And saturated pressure

When tests evaluate status handling

Then the Provider is not classified as failed solely because it is saturated.

---

### Requirement: Provider Loading Failure Tests

Dynamic Provider loading SHALL be tested for failure modes.

#### Scenario: Unsupported ABI version

Given a dynamic Provider fixture reports an unsupported ABI version

When Runtime attempts loading

Then tests verify loading fails before Provider registration.

---

### Requirement: Provider Refusal Tests

Provider refusal before execution SHALL be tested separately from execution
failure after submission.

#### Scenario: Provider rejects admission

Given a Provider refuses work because it is draining

When Runtime attempts submission

Then tests verify the error is classified as admission/status failure.

---

### Requirement: Provider Drain Tests

Provider drain behavior SHALL be covered by tests.

#### Scenario: New work during drain

Given a Provider is draining

When new unpinned work is submitted

Then tests verify the Provider is not selected.

---

### Requirement: Provider Resource Ownership Tests

Provider-owned resource behavior SHALL be tested.

#### Scenario: Bound tensor

Given a tensor is Provider-bound

When dependent work is planned

Then tests verify the Runtime preserves binding or requires explicit movement.
