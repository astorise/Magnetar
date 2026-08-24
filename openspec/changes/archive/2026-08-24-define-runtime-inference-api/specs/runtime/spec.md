## ADDED Requirements

### Requirement: Runtime Owns Inference API Validation

Runtime SHALL validate all Runtime Inference API requests before execution.

#### Scenario: Invalid request

Given generation request references unknown session

When Runtime validates it

Then Runtime rejects it with session-not-found.

---

### Requirement: Runtime Inference API Does Not Grant External Authority

Runtime Inference API SHALL not grant filesystem, network, process, shell, Git, secret, workspace, tool, or external-service authority.

#### Scenario: Tool call requested

Given caller asks Runtime to execute an external tool

When Runtime validates inference request

Then request is rejected as outside Runtime scope.

---

### Requirement: Runtime Preserves Internal Boundaries

Runtime Inference API SHALL route through Model Loading, Model Instance, Tokenizer, Session, Generation, Sampling, Memory Manager, Kernel Registry, Provider, and Observability contracts instead of bypassing them.

#### Scenario: One-shot inference

Given one-shot request is submitted

When Runtime executes it

Then Runtime still creates an implicit session and uses normal inference
contracts.