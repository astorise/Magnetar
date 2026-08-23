## ADDED Requirements

### Requirement: Provider Conformance Suite

Magnetar SHALL define a Provider Conformance Suite that validates Provider
implementations against the Provider contract.

#### Scenario: Run conformance suite

Given a Provider implementation is supplied as a conformance target

When the suite runs required profiles

Then the suite reports whether the Provider conforms to Magnetar's Provider
contract.

---

### Requirement: Conformance Profiles

The conformance suite SHALL support profiles corresponding to Provider
features.

Initial profiles SHOULD include:

- provider-core
- provider-compute
- provider-data-movement
- provider-cancellation
- provider-observability
- provider-dynamic-abi

#### Scenario: Provider advertises Compute

Given a Provider advertises Compute Capability

When conformance is evaluated

Then the Provider must pass the provider-compute profile.

---

### Requirement: Provider Core Conformance

The `provider-core` profile SHALL validate metadata, identity, Capability
advertisements, Device metadata, lifecycle, status, error mapping, and basic
Runtime registration behavior.

#### Scenario: Invalid metadata

Given a Provider reports invalid Provider metadata

When the core conformance profile runs

Then the Provider fails conformance.

---

### Requirement: Capability Advertisement Conformance

A Provider SHALL not pass conformance if its advertised Capability support does
not match observed behavior.

#### Scenario: Advertised unsupported operation

Given a Provider advertises support for operation X

But rejects every valid operation X request as unsupported

When conformance runs

Then the Provider fails the relevant profile.

---

### Requirement: Device Metadata Conformance

Provider Device metadata SHALL be validated during conformance.

Device metadata SHALL not expose raw native handles as public metadata.

#### Scenario: Raw native handle exposed

Given Device metadata includes a raw CUDA or driver handle as public metadata

When conformance runs

Then the Provider fails conformance.

---

### Requirement: Provider Status Conformance

Provider status reporting SHALL conform to the refined status model.

The suite SHALL validate lifecycle, health, readiness, pressure, admission,
freshness, Device status, and Capability status where applicable.

#### Scenario: Saturated reported as failed

Given a Provider is saturated but internally healthy

When conformance evaluates status reporting

Then reporting saturation as generic failure fails conformance.

---

### Requirement: Compute Conformance

A Provider advertising Compute Capability SHALL pass Compute conformance tests.

Compute conformance SHALL include valid execution, invalid request rejection,
unsupported operation rejection, output descriptor validation, and stable error
mapping.

#### Scenario: Invalid dtype

Given a Compute request uses an unsupported dtype

When the Provider handles the request

Then the Provider returns the stable unsupported-dtype error expected by
Magnetar.

---

### Requirement: Numerical Correctness

Providers performing numerical operations SHALL meet declared numerical
tolerance for conformance fixtures.

#### Scenario: Approximate dtype

Given a Provider declares approximate numeric behavior for a dtype

When conformance validates output

Then output must fall within the accepted tolerance for that declaration.

---

### Requirement: Data Movement Conformance

A Provider advertising data movement support SHALL pass data movement
conformance tests for each advertised movement type.

#### Scenario: Advertised upload

Given a Provider advertises upload support

When conformance performs a valid upload fixture

Then the Provider completes it according to Magnetar data movement semantics.

---

### Requirement: Resource Affinity Conformance

Provider-owned and Device-bound resources SHALL obey Resource Affinity
requirements.

#### Scenario: Device-bound resource

Given a resource is bound to Device A

When dependent work is submitted

Then the Provider preserves the Device binding or reports a structured
incompatibility requiring explicit movement.

---

### Requirement: Cancellation Conformance

A Provider advertising cancellation support SHALL pass cancellation conformance
tests.

A Provider not supporting cancellation SHALL report unsupported cancellation
through stable errors.

#### Scenario: Cancellation unsupported

Given a Provider does not support cancellation

When cancellation is requested

Then it reports the stable cancellation-unsupported error.

---

### Requirement: Error Mapping Conformance

Provider errors SHALL map to stable Magnetar error categories.

#### Scenario: Out of memory

Given a Provider cannot allocate required memory

When execution is attempted

Then the Provider reports a stable out-of-memory or allocation-failure category

And does not return an opaque implementation-specific failure only.

---

### Requirement: Observability Conformance

Provider operations SHALL integrate with Runtime observability without making
observability authoritative for execution.

#### Scenario: Observation sink fails

Given Provider execution succeeds

And observability delivery fails

When conformance evaluates the result

Then execution remains successful.

---

### Requirement: Dynamic ABI Conformance

Dynamic Providers SHALL pass dynamic ABI conformance tests.

#### Scenario: Missing release function

Given a dynamic Provider ABI descriptor lacks a required release function

When dynamic ABI conformance runs

Then the Provider fails conformance.

---

### Requirement: Non-Conformant Provider Is Not Production Compatible

A Provider failing required conformance profiles SHALL NOT be documented or
registered as production-compatible.

#### Scenario: Provider fails core profile

Given a Provider fails the provider-core profile

When compatibility status is reported

Then the Provider is marked non-conformant.

---

### Requirement: Conformance Report

The conformance suite SHALL produce a structured report.

The report SHOULD include Provider identity, Provider version, Runtime version,
suite version, selected profiles, passed tests, failed tests, skipped tests, and
diagnostics.

#### Scenario: Report generated

Given conformance completes

When the report is emitted

Then it can be consumed by humans and automated CI.
