## ADDED Requirements

### Requirement: First Operator Scope Release Gate

First operator scope conformance SHALL be required for `v0.1`.

#### Scenario: Required operator missing

Given required operator coverage is missing

When release validation runs

Then stable release is blocked.

---

### Requirement: Operator Compatibility Status

Release matrix SHALL document Operator catalog compatibility status.

#### Scenario: Operator status

Given release notes are generated

When Operator catalog is listed

Then its status is declared.