## ADDED Requirements

### Requirement: E2E Uses Execution Graph Contract

E2E conformance SHALL validate Execution Graph production, validation, planning,
and execution through Runtime.

#### Scenario: Prefill graph

Given Qwen baseline produces prefill graph

When Runtime validates it

Then graph uses portable Operators and valid Tensor metadata.

---

### Requirement: E2E Detects Invalid Graphs

E2E conformance SHALL include invalid graph failure cases.

#### Scenario: Invalid tensor edge

Given graph fixture contains invalid tensor edge

When Runtime validates it

Then graph validation fails.