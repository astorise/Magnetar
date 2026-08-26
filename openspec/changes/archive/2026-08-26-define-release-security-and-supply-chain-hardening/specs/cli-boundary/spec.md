## ADDED Requirements

### Requirement: CLI Release Authority Boundary

Release CLI SHALL not delegate ambient filesystem, Git, network, secret, shell,
process, tool, workspace, or agent authority to Runtime.

#### Scenario: CLI reads file

Given CLI reads file for prompt

When Runtime request is created

Then Runtime receives explicit prompt/context only.

---

### Requirement: CLI Release Secret Redaction

CLI release diagnostics and observations SHALL redact secrets by default.

#### Scenario: Secret provider error

Given CLI fails to read secret

When diagnostic is emitted

Then secret value is absent.