## ADDED Requirements

### Requirement: Release Conformance Gates

Stable release SHALL pass required conformance gates.

#### Scenario: Provider conformance failure

Given Reference CPU conformance fails

When release is attempted

Then stable release is blocked.

---

### Requirement: Conformance Reports Included In Release

Release artifacts SHALL include conformance reports where applicable, or SHALL
explicitly mark them not applicable.

#### Scenario: Release artifact check

Given release candidate is prepared

When artifacts are inspected

Then conformance report is present or explicitly not applicable.

---

### Requirement: Conformance Suite Versions In Release

Release metadata SHALL include conformance suite versions.

#### Scenario: Report metadata

Given E2E report is generated

When release metadata is assembled

Then E2E suite version is included.