## ADDED Requirements

### Requirement: Qualification Coverage For Specialization

Qualification Record SHALL be able to express specialization coverage.

#### Scenario: Exact variant

Given qualification tested only tile size 64

When tile size 32 is considered

Then evidence does not cover it unless explicitly declared.

---

### Requirement: Envelope Qualification Must Be Explicit

Specialization envelope MAY inherit qualification only when qualification profile/evidence explicitly authorizes it, and Runtime SHALL reject envelope coverage lacking explicit qualification authorization.

#### Scenario: All allowed warp counts proven

Given qualification establishes correctness for warp counts {4,8}

When either specialization is used

Then evidence MAY cover both.

---

### Requirement: Autotuning Cannot Expand Qualification

Autotuning benchmark SHALL not widen qualification envelope.

#### Scenario: Untested candidate benchmarks correctly once

Given candidate is outside qualification coverage

When benchmark produces plausible output

Then it remains outside qualification coverage.