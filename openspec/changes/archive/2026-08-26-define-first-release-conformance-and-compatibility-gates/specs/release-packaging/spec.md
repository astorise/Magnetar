## ADDED Requirements

### Requirement: Packaging Depends On Release Gates

Release packaging SHALL depend on successful release conformance gates.

#### Scenario: Packaging attempted early

Given required gates have not passed

When stable packaging is attempted

Then stable release packaging is blocked.

---

### Requirement: Release Reports Packaged

Release packaging SHALL include machine-readable and human-readable gate
reports where release gates have run.

#### Scenario: Release artifact

Given release artifacts are assembled

When inspected

Then conformance and compatibility reports are present or explicitly not
applicable.