## ADDED Requirements

### Requirement: Observability Release Redaction Gate

Observability redaction gate SHALL be required for `v0.1`.

#### Scenario: Secret logged

Given observation logs secret by default

When release validation runs

Then stable release is blocked.

---

### Requirement: Release Reports Redacted

Release reports SHALL be redacted by default.

#### Scenario: Report failure

Given failure includes prompt text

When release report is generated

Then raw prompt text is absent by default.