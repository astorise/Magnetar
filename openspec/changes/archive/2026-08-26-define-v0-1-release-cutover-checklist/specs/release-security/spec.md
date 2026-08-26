## ADDED Requirements

### Requirement: Security Cutover Required

Release cutover SHALL verify release security before publication.

#### Scenario: Security notes missing

Given security notes are missing

When cutover runs

Then release is blocked.

---

### Requirement: Security Status Published

Release notes SHALL publish security status and limitations.

#### Scenario: Signatures unavailable

Given signatures are unavailable

When release notes are published

Then limitation is stated.