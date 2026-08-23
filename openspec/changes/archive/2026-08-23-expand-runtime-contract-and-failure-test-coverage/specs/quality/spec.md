## ADDED Requirements

### Requirement: Failure-Oriented Test Coverage

The repository SHALL include tests for critical Runtime failure modes, not only
successful paths.

#### Scenario: Provider refuses work

Given a Provider is compatible but not ready

When Runtime attempts to use it

Then tests verify that the Runtime returns a structured status/admission error.

---

### Requirement: Architecture Invariant Tests

The repository SHALL include tests that protect core Magnetar architecture
invariants.

#### Scenario: Component cannot select Provider

Given a Component-facing Compute request

When its request structure is validated

Then no Provider selector is accepted.

---

### Requirement: Contract Tests

Contract-level tests SHALL validate externally visible Runtime behavior across
module boundaries.

#### Scenario: Runtime resolves Provider

Given two mock Providers advertise the same Capability

When a Component requests that Capability

Then Runtime resolution behavior matches Resource Affinity and Resolution
Policy.

---

### Requirement: Failure Injection Utilities

The repository SHALL provide utilities or fixtures for injecting controlled
Runtime failures.

#### Scenario: Saturated Provider

Given a mock Provider is configured as saturated

When Resolution evaluates it

Then the Runtime applies the configured saturated-provider policy.

---

### Requirement: Feature-Gated Test Coverage

When an important implementation is feature-gated, CI SHALL include at least
one job enabling the feature and running its tests.

#### Scenario: Wasmtime feature

Given the concrete WASM Component engine is feature-gated

When CI runs

Then at least one CI job enables it and runs Component execution tests.

---

### Requirement: Coverage Measures Meaningful Behavior

Coverage improvements SHALL focus on meaningful Runtime behavior and failure
paths.

Tests that only execute trivial getters, constructors, or formatting code SHALL
not be treated as sufficient coverage for critical contracts.

#### Scenario: Low coverage Provider module

Given Provider loading has low coverage

When coverage is expanded

Then tests cover successful loading and failure modes rather than only simple
metadata accessors.

---

### Requirement: Deterministic Fixtures

Test fixtures SHALL be deterministic and suitable for local and CI execution.

Fixtures SHALL NOT require external network access, Tachyon, or real GPU
hardware unless explicitly marked as optional.

#### Scenario: Run tests offline

Given CI runs without external network

When the test suite executes

Then required fixture tests still pass.

---

### Requirement: Coverage Ratchet Preservation

This change SHALL not lower established coverage thresholds.

New coverage targets MAY be raised when meaningful test coverage improves.

#### Scenario: Add failure tests

Given new failure tests improve coverage

When CI thresholds are updated

Then thresholds may increase but are not reduced to accommodate untested code.
