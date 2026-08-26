## ADDED Requirements

### Requirement: Runtime Cutover Scope Confirmation

Cutover SHALL confirm Runtime remains inference-only in `v0.1`.

#### Scenario: Runtime includes tool execution

Given Runtime API includes tool execution

When cutover validates scope

Then release is blocked.

---

### Requirement: Runtime Version Matches Release

Runtime version metadata SHALL match release tag or documented version mapping.

#### Scenario: Version mismatch

Given Runtime reports `0.1.0-rc.1`

When stable `v0.1.0` release is published

Then cutover verification fails.