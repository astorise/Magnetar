## ADDED Requirements

### Requirement: Conformance Suite Release Mode

Conformance SHALL support release mode for `v0.1` gates.

#### Scenario: Release conformance run

Given release mode is enabled

When conformance executes

Then required baseline suites are run and optional out-of-scope suites are
skipped with reasons.

---

### Requirement: Conformance Report Redaction

Conformance reports SHALL be redacted by default.

#### Scenario: Failure report

Given conformance failure involves prompt input

When report is generated

Then raw prompt text is absent by default.