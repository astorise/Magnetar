## ADDED Requirements

### Requirement: Optimization Recommendation Is Selection Input Only

Optimization recommendation MAY inform Runtime Kernel Selection Policy, but it SHALL NOT override eligibility.

#### Scenario: Recommended but memory infeasible

Given optimizer recommends candidate

When Memory Manager rejects its workspace

Then candidate remains ineligible.

---

### Requirement: Production Ranking Uses Current Context

Runtime SHALL use current policy/context rather than blindly replaying
optimization campaign ranking.

#### Scenario: Device pressure changed

Given campaign ranked GPU candidate first

But production GPU is unavailable

When Runtime selects Kernel

Then campaign ranking does not force unavailable candidate.

---

### Requirement: Promotion Request Uses Normal Selection Rules

Candidate submitted by Optimization Plane SHALL pass normal Kernel selection
eligibility.

#### Scenario: Trust revoked

Given candidate recommendation exists

When trust is revoked before promotion

Then selection excludes it.