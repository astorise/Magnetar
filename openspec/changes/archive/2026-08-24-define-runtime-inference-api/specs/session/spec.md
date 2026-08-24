## ADDED Requirements

### Requirement: Sessions Are Exposed Through Inference API

Runtime Inference API SHALL expose Inference Session creation, lookup, use, and closure.

#### Scenario: Create session

Given a Model Instance is ready

When caller creates session

Then Runtime creates a Runtime-owned Inference Session.

---

### Requirement: Inference API Sessions Are Not Agent Sessions

Inference Sessions exposed through Runtime Inference API SHALL not own agent, tool, workspace, Git, shell, network, or secret state.

#### Scenario: Agent state requested

Given caller tries to attach tool state to Inference Session

When Runtime validates request

Then Runtime rejects it as outside inference scope.