## ADDED Requirements

### Requirement: Reference CPU Provider Implemented Before E2E

Reference CPU Provider baseline SHALL be implemented before E2E local inference
success path.

#### Scenario: CPU execution

Given required-now operators are used

When E2E runs

Then Reference CPU Provider supplies compatible kernels.

---

### Requirement: Reference CPU Prioritizes Correctness

Reference CPU implementation SHALL prioritize deterministic correctness over
performance for the baseline.

#### Scenario: Slow attention

Given CPU attention is slow

When output is correct for fixture

Then baseline acceptance may pass.