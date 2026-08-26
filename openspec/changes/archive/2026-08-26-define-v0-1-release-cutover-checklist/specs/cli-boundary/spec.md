## ADDED Requirements

### Requirement: CLI Boundary Cutover Verification

Cutover SHALL verify CLI/Runtime boundary gates before publication.

#### Scenario: CLI boundary gate failed

Given Runtime receives ambient CLI filesystem authority

When cutover runs

Then release is blocked.

---

### Requirement: CLI Surface Status In Release Notes

Release notes SHALL declare CLI command surface status.

#### Scenario: CLI preview

Given `magnetar run` exists but is preview

When release notes are published

Then it is marked preview or stable-for-v0.1-baseline accurately.