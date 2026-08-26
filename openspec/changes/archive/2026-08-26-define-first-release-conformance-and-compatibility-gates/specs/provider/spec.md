## ADDED Requirements

### Requirement: Provider Release Gate Coverage

Provider contracts SHALL have release gate coverage.

#### Scenario: Provider health

Given Provider health/readiness/pressure are conflated

When release gate runs

Then release is blocked.

---

### Requirement: Non-Baseline Providers Are Skippable With Reason

A skip SHALL include a structured out-of-scope reason. Non-baseline Providers
MAY be skipped only with such a reason.

#### Scenario: CUDA skip

Given CUDA is not in `v0.1`

When release report is generated

Then CUDA gate is skipped with reason.