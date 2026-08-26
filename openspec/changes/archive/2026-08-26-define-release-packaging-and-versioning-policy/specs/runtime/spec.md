## ADDED Requirements

### Requirement: Runtime Release Version Metadata

Runtime SHALL expose release version metadata where appropriate.

#### Scenario: Runtime diagnostics

Given diagnostics are requested

When version metadata is included

Then Runtime version and OpenSpec baseline version are available.

---

### Requirement: Runtime Release Does Not Expand Scope Silently

Runtime release SHALL not include additional responsibilities such as workspace,
Git, shell, process, tools, secrets, or agent orchestration.

#### Scenario: Release audit

Given `v0.1` release is audited

When Runtime public API is inspected

Then inference-only scope is preserved.