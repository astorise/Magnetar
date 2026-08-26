## ADDED Requirements

### Requirement: CLI Boundary Release Gate

CLI boundary conformance SHALL be required for `v0.1`.

#### Scenario: Ambient authority leak

Given CLI filesystem authority is delegated to Runtime

When release gate runs

Then stable release is blocked.

---

### Requirement: CLI Compatibility Status

CLI command surface compatibility status SHALL appear in release matrix.

#### Scenario: CLI matrix

Given `magnetar run` exists

When release notes are generated

Then CLI command status is marked stable-for-v0.1-baseline, preview,
experimental, unstable, deferred, or unsupported.