## ADDED Requirements

### Requirement: Source Cache Conformance

Conformance SHALL validate model source and cache behavior.

#### Scenario: Cache hit validation

Given cached artifact exists

When conformance loads it

Then trust, integrity, format, and loading validations still run.

---

### Requirement: Source Cache Boundary Conformance

Conformance SHALL validate Runtime does not gain arbitrary filesystem, network,
credential, or cache mutation authority.

#### Scenario: Arbitrary directory scan

Given Runtime is asked to scan arbitrary model directory

When conformance runs

Then request is denied.

---

### Requirement: Cache Residency Conformance

Conformance SHALL validate cache presence is distinct from memory residency.

#### Scenario: Cached but not loaded

Given artifact is cached but not loaded

When Memory Manager is inspected

Then no model tensors are resident.