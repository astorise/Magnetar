## ADDED Requirements

### Requirement: Memory Manager Governs Autotuning Workspace

Autotuning candidate SHALL not benchmark with memory resources unavailable to
normal production policy.

#### Scenario: Fast variant needs excessive workspace

Given workspace exceeds allowed memory

When tuning plans candidate

Then candidate is rejected or skipped as production-infeasible.

---

### Requirement: Tuning Allocations Are Temporary

Autotuning-specific tensor/workspace allocations SHALL be released after tuning
lifecycle.

#### Scenario: Benchmark finishes

Given candidate benchmark used temporary buffers

When candidate measurement completes

Then buffers are reclaimed according to Runtime policy.

---

### Requirement: Tuning Cannot Steal Unbounded Inference Memory

Runtime SHALL bound memory consumed by tuning.

#### Scenario: Active model under pressure

Given tuning budget would violate reserved inference capacity

When admission occurs

Then tuning is denied/postponed.