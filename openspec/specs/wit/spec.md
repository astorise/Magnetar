# wit Specification

## Purpose
TBD - created by archiving change define-release-packaging-and-versioning-policy. Update Purpose after archive.
## Requirements
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

### Requirement: WIT Cutover Version Confirmation

Cutover SHALL confirm included WIT package versions before stable release.

#### Scenario: Missing WIT matrix

Given WIT package exists but is missing from compatibility matrix

When cutover runs

Then release is blocked.

---

### Requirement: WIT Cutover Gate Before Tag

WIT validation SHALL complete before stable release tag.

#### Scenario: Tag before WIT validation

Given stable tag is created before WIT validation

When cutover validates sequence

Then release is invalid.

