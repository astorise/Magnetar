## ADDED Requirements

### Requirement: Local Inference Conformance Suite

Conformance SHALL include a local inference suite that validates the full
Runtime inference path.

#### Scenario: Run local suite

Given conformance is executed

When local inference suite runs

Then the suite validates complete Runtime inference behavior.

---

### Requirement: E2E Conformance Uses Normal Runtime Contracts

E2E conformance SHALL use normal Runtime contracts and SHALL NOT use hidden
shortcuts.

#### Scenario: Shortcut detected

Given test path bypasses Model Loading

When conformance validates the path

Then the suite fails.

---

### Requirement: E2E Conformance Report

Conformance SHALL include E2E report output in machine-readable form.

#### Scenario: Report included

Given E2E suite completes

When conformance results are collected

Then E2E report is included with structured pass/fail/skipped status.