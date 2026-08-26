## ADDED Requirements

### Requirement: Reference CPU Release Gate Required

Reference CPU Provider gate SHALL be required for `v0.1`.

#### Scenario: Reference CPU absent

Given Reference CPU Provider is unavailable

When release validation runs

Then stable release is blocked.

---

### Requirement: Reference CPU Correctness Report

Release report SHALL include Reference CPU conformance status.

#### Scenario: CPU report

Given conformance runs

When release report is generated

Then Reference CPU status is included.