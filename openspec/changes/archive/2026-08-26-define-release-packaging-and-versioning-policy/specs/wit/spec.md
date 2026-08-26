## ADDED Requirements

### Requirement: WIT Package Release Versions

WIT packages SHALL have explicit release versions.

#### Scenario: WIT released

Given WIT package is included in release

When metadata is inspected

Then package version is declared.

---

### Requirement: WIT Breaking Change Policy

Breaking WIT changes SHALL require a major version bump.

#### Scenario: Interface removed

Given WIT interface is removed

When release validation runs

Then major version bump is required.

---

### Requirement: WIT Supported Version Matrix

Release documentation SHALL include supported WIT versions.

#### Scenario: Runtime supports compute WIT v2

Given release docs are generated

When WIT support is inspected

Then supported package versions are listed.