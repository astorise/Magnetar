## ADDED Requirements

### Requirement: CLI Boundary Conformance

Conformance SHALL validate that `magnetar-cli` and Runtime preserve the
inference boundary.

#### Scenario: File access boundary

Given CLI reads file content for prompt

When Runtime receives request

Then Runtime has no filesystem authority.

---

### Requirement: Runtime Does Not Execute CLI-Owned Capabilities

Conformance SHALL validate Runtime does not execute tools, shell, Git, network,
or workspace operations.

#### Scenario: Generated shell text

Given model output contains shell command text

When Runtime emits output

Then no process execution occurs.

---

### Requirement: CLI Preserves Runtime Structured Errors

Conformance SHALL validate CLI preserves Runtime structured error categories
when displaying or wrapping errors.

#### Scenario: Runtime model loading error

Given Runtime returns model-loading-failed

When CLI displays failure

Then structured category is preserved.