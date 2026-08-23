# resolution Specification

## Purpose
TBD - created by archiving change refine-provider-health-readiness-pressure-model. Update Purpose after archive.
## Requirements
### Requirement: Resolution Considers Provider Status

Resolution SHALL consider Provider lifecycle, health, readiness, pressure,
admission, and status freshness.

#### Scenario: Provider not ready

Given a Provider implements the requested Capability

But readiness is not-ready

When Resolution evaluates candidates

Then the Provider is not selected for ordinary new work.

---

### Requirement: Resolution Filters Failed Providers

Resolution SHALL reject Providers in failed health or failed lifecycle state for
new work.

#### Scenario: Failed Provider

Given a Provider advertises the requested Capability

And its health is failed

When Resolution evaluates candidates

Then the Provider is rejected.

---

### Requirement: Resolution Handles Degraded Providers By Policy

Resolution SHALL handle degraded Provider eligibility according to Runtime
policy.

#### Scenario: Degraded Provider allowed

Given a Provider is degraded

And policy allows degraded Providers for low-priority work

When compatible low-priority work is resolved

Then the Provider may remain eligible with a penalty.

---

### Requirement: Resolution Handles Pressure By Policy

Provider pressure SHALL influence candidate ranking or rejection according to
policy.

#### Scenario: Low pressure preferred

Given two compatible Providers are ready

And one has low pressure while the other has high pressure

When Resolution ranks candidates

Then policy may prefer the lower-pressure Provider.

---

### Requirement: Resolution Rejects Saturated Provider Unless Queuing Allowed

A saturated Provider SHALL be rejected for ordinary new work unless policy
explicitly allows queueing or delayed admission.

#### Scenario: Saturated Provider

Given a Provider is saturated

And policy does not allow queueing

When Resolution evaluates candidates

Then the Provider is rejected.

---

### Requirement: Resolution Avoids Draining Provider For New Work

A draining Provider SHALL NOT be selected for ordinary new unpinned work.

#### Scenario: New unpinned request

Given Provider A is draining

And Provider B is compatible and ready

When a new unpinned request is resolved

Then Provider B is preferred or selected.

---

### Requirement: Resource Affinity Overrides Provider Status Preference

Resource Affinity SHALL remain authoritative.

Provider status preference SHALL NOT silently move Provider-bound resources.

#### Scenario: Bound resource on draining Provider

Given a resource is Provider-pinned to Provider A

And Provider A is draining

When dependent work requires the pinned resource

Then Runtime policy determines whether compatible pinned work may continue

And Resolution does not silently select Provider B without explicit movement.

---

### Requirement: Resolution Treats Stale Status Conservatively

If Provider status is stale, Resolution SHALL apply policy.

The default SHOULD be to reject or penalize stale Providers for new work.

#### Scenario: Stale Provider report

Given a Provider's status TTL has expired

When Resolution evaluates candidates

Then the Provider is not treated as fully ready by default.

---

### Requirement: Resolution Uses Device-Level Status

Resolution SHALL consider Device-level status when selecting execution
placement.

#### Scenario: Device unavailable

Given a Provider is healthy

But the target Device is unavailable

When Resolution selects placement

Then that Device is rejected.

---

### Requirement: Resolution Uses Capability-Level Status

Resolution SHALL consider Capability-level status.

#### Scenario: Capability not ready

Given a Provider is ready for one Capability

But not ready for the requested Capability

When Resolution evaluates candidates

Then the Provider is rejected or delayed for that request.

---

### Requirement: Resolution Produces Status Diagnostics

Resolution diagnostics SHALL explain status-based candidate decisions.

#### Scenario: Provider skipped due to pressure

Given a Provider is skipped because pressure is saturated

When diagnostics are returned

Then the diagnostic includes a stable saturated-provider reason.

