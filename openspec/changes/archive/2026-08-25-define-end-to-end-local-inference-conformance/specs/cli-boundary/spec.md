## ADDED Requirements

### Requirement: E2E Validates CLI Boundary

E2E conformance SHALL validate that CLI-owned authorities are not delegated to
Runtime.

#### Scenario: File context

Given CLI harness reads file content for prompt

When Runtime receives inference request

Then Runtime receives explicit prompt/context and no filesystem authority.

---

### Requirement: E2E Validates Runtime Does Not Execute Tools

E2E conformance SHALL validate Runtime does not execute tools, shell, Git,
network, or workspace operations.

#### Scenario: Generated tool text

Given Runtime output contains tool-call-like text

When E2E verifies effects

Then no Runtime-side tool execution occurred.