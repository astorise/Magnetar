## ADDED Requirements
### Requirement: Registry Resolution May Feed Plan Construction

Kernel Registry SHALL expose candidate metadata needed to construct Prepared
Execution Plan.

#### Scenario: Plan build

Given three eligible MatMul candidates

When Plan construction runs

Then Runtime selection policy resolves one candidate and records exact binding.

---

### Requirement: Ready Plan Avoids Repeated Full Registry Resolution

Execution of compatible ready Plan SHALL not require complete Registry
candidate discovery on every invocation.

#### Scenario: Repeated decode

Given same compatible Plan remains ready

When successive token steps execute

Then node bindings can reuse Plan decisions.

---

### Requirement: Registry Change Does Not Mutate Existing Plan

Kernel Registry preference change SHALL not rewrite an already acquired Plan
generation.

#### Scenario: Kernel v3 promoted

Given active Plan references v2

When Registry changes preference to v3

Then existing Plan becomes stale/replacement candidate rather than changing
binding in place.

---

### Requirement: Revocation Propagates To Dependent Plans

Registry/revocation system SHALL permit Runtime to identify Plans requiring a
revoked Kernel.

#### Scenario: Security revocation

Given Kernel digest is revoked

When Runtime evaluates dependent Plans

Then they become invalid for new work.
