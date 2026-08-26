## ADDED Requirements

### Requirement: WIT Release Gate

Included WIT contracts SHALL have release gate coverage.

#### Scenario: WIT version missing

Given included WIT package has no version

When release validation runs

Then stable release is blocked.

---

### Requirement: WIT Compatibility Matrix Entry

Release compatibility matrix SHALL include WIT package statuses.

#### Scenario: WIT matrix

Given release notes are generated

When WIT packages are listed

Then each included package has a compatibility status.