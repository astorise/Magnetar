## ADDED Requirements

### Requirement: Reference CPU Kernels Enter Registry Through Validation

Reference CPU Kernel advertisements SHALL be validated before registry
insertion.

#### Scenario: Invalid CPU advertisement

Given Reference CPU Provider advertises unknown Operator

When Runtime validates it

Then the advertisement is rejected.

---

### Requirement: Reference CPU Candidate Selection

Reference CPU Kernels SHALL participate in normal Kernel candidate lookup,
filtering, ranking, fallback, and dispatch.

#### Scenario: CPU candidate

Given graph contains matmul

When Kernel Registry queries candidates

Then Reference CPU matmul may be considered if advertised and policy allows.

---

### Requirement: Reference CPU Fallback Observable

Fallback to Reference CPU SHALL be explicit and observable.

#### Scenario: CPU fallback used

Given optimized Kernel is unavailable

And policy permits CPU fallback

When Runtime selects Reference CPU

Then observability records fallback usage.