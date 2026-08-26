## ADDED Requirements

### Requirement: Runtime Release Gate Coverage

Runtime SHALL have release gate coverage for inference-only authority, boundary
rejection, error structure, and redaction.

#### Scenario: Runtime file authority

Given Runtime exposes arbitrary file read

When release gate runs

Then release is blocked.

---

### Requirement: Runtime Release Compatibility Status

Runtime release metadata SHALL mark Runtime compatibility status.

#### Scenario: Runtime status

Given release notes are generated

When Runtime status is listed

Then it is marked stable-for-v0.1-baseline, preview, experimental, unstable,
deferred, or unsupported.