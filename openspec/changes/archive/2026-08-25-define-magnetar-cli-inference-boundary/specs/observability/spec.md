## ADDED Requirements

### Requirement: CLI And Runtime Observability Are Distinct

CLI-side observations and Runtime-side observations SHALL remain distinct.

#### Scenario: Command plus inference

Given CLI runs `magnetar run`

When observations are emitted

Then CLI may observe command parsing while Runtime observes inference execution.

---

### Requirement: CLI Observability Redacts Sensitive Context

CLI observability SHALL redact raw prompts, secrets, file contents, tokens,
model weights, handles, and memory pointers by default.

#### Scenario: File prompt

Given CLI reads file content for prompt

When CLI emits observations

Then raw file content is not logged by default.

---

### Requirement: Runtime Observability Does Not Log CLI Authority

Runtime observations SHALL not log CLI authority, workspace permissions, secret
providers, network credentials, or tool capabilities.

#### Scenario: CLI has tool access

Given CLI has tool access

When Runtime emits inference observations

Then Runtime does not log tool capability details unless explicitly included as
redacted request metadata.