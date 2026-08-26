## ADDED Requirements

### Requirement: Serve Mode Preserves CLI Runtime Boundary

If `magnetar serve` is implemented in CLI, it SHALL still call Runtime Inference
API and preserve CLI/Runtime authority boundary.

#### Scenario: CLI serve

Given CLI starts serve mode

When request needs inference

Then serve mode calls Runtime Inference API with explicit request data.

---

### Requirement: Serve Mode Does Not Become Agent Runtime

Core serve mode SHALL not own agent planning, tool execution, workspace mutation,
Git, shell, or external service orchestration.

#### Scenario: Generated tool call

Given server streams generated tool-call-like text

When core serve mode handles it

Then it does not execute the tool.