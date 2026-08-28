## ADDED Requirements

### Requirement: Registry Provides Candidate Set

Kernel Registry SHALL expose candidate metadata to Runtime selection policy
without performing opaque policy ranking itself.

#### Scenario: Multiple MatMul kernels

Given Registry contains four compatible MatMul implementations

When selection begins

Then Runtime policy can evaluate candidates explicitly.

---

### Requirement: Registry Eligibility Metadata

Registry candidate metadata SHALL include information required for eligibility
evaluation.

#### Scenario: Qualified candidate

Given Registry returns candidate

When Runtime evaluates it

Then qualification, trust, target and preparation state are available.

---

### Requirement: Registry Does Not Make Cross-Provider Optimization Decision

Kernel Registry SHALL not independently choose the globally fastest Provider
without Runtime selection policy.

#### Scenario: CPU and CUDA candidates

Given both exist

When Kernel is selected

Then Runtime policy decides according to eligibility and objective.

---

### Requirement: Registry Supports Stable Candidate Identity

Candidate identity SHALL be stable enough for deterministic tie-breaking.

#### Scenario: Equal benchmark scores

Given candidates tie

When stable key is compared

Then deterministic ordering is available.

---

### Requirement: Registry Respects Revocation

Revoked candidate SHALL be excluded before optimization ranking.

#### Scenario: Previously fastest Kernel revoked

Given fastest candidate becomes revoked

When Registry candidates are evaluated

Then it cannot be selected.