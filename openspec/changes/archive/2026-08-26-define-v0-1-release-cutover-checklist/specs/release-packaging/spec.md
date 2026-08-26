## ADDED Requirements

### Requirement: Packaging Follows Cutover Checklist

Release packaging SHALL follow the `v0.1` cutover checklist.

#### Scenario: Packaging before gates

Given required gates have not passed

When packaging is attempted

Then stable release packaging is blocked.

---

### Requirement: Packaging Includes Cutover Evidence

Release packaging SHALL include or reference cutover evidence such as reports,
checksums, compatibility matrix, security notes, and release notes.

#### Scenario: Artifact bundle

Given release artifact bundle is assembled

When inspected

Then cutover evidence is available.