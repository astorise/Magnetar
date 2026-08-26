## ADDED Requirements

### Requirement: Runtime Has No CLI Ambient Authority

Runtime SHALL not inherit CLI filesystem, network, Git, secret, shell, process,
tool, workspace, or agent authority.

#### Scenario: CLI authorized network

Given CLI has network authorization

When Runtime receives inference request

Then Runtime still has no arbitrary network authority.

---

### Requirement: Runtime Rejects CLI Boundary Violations

Runtime SHALL reject requests that attempt to use Runtime as workspace, Git,
network, secret, tool, process, shell, or agent runtime.

#### Scenario: Workspace scan request

Given request asks Runtime to scan workspace

When Runtime validates request

Then Runtime rejects it as outside inference scope.

---

### Requirement: Runtime Accepts Prepared Prompt Context

Runtime SHALL accept prepared inference input from CLI according to Runtime
Inference API contracts.

#### Scenario: Prompt context

Given CLI sends prompt/context text

When Runtime validates request

Then Runtime treats it as inference input and not as authority.