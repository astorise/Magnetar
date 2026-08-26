## ADDED Requirements

### Requirement: CLI Release Boundary

Release CLI surface SHALL preserve CLI/Runtime boundary.

#### Scenario: CLI command

Given released CLI invokes inference

When execution is traced

Then CLI calls Runtime Inference API.

---

### Requirement: CLI Command Stability Status

Release metadata SHALL declare CLI command surface compatibility status.

#### Scenario: CLI docs

Given `magnetar run` is documented

When release notes are inspected

Then command is marked stable-for-baseline, experimental, or preview.