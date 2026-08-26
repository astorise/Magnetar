## ADDED Requirements

### Requirement: Packaging Includes Security Artifacts

Release packaging SHALL document security release status, and SHOULD include
security-related artifacts such as checksums, SBOM or SBOM limitation,
provenance, security notes, and audit status.

#### Scenario: Release package

Given release artifacts are assembled

When package metadata is inspected

Then security artifact status is present.

---

### Requirement: Stable Release Blocked By Security Gate Failure

Release packaging SHALL not publish stable release when required security gates
fail.

#### Scenario: Redaction gate fails

Given redaction gate fails

When release publication is attempted

Then stable publication is blocked.