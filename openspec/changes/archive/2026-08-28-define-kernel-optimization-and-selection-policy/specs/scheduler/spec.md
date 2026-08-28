## ADDED Requirements

### Requirement: Scheduler Supplies Workload Context

Runtime SHALL treat Scheduler-provided workload state as an optimization input rather than an eligibility constraint; Scheduler MAY provide such state.

#### Scenario: Continuous batch

Given batch contains 32 active sequences

When throughput selection occurs

Then batch width may influence ranking.

---

### Requirement: Scheduler Does Not Override Kernel Eligibility

Scheduler SHALL not force an ineligible Kernel for throughput.

#### Scenario: Batch favors GPU

Given GPU Kernel violates affinity

When Scheduler wants throughput

Then Runtime still excludes GPU Kernel.

---

### Requirement: Scheduler Does Not Own Kernel Policy

Scheduler MAY provide load/queue context, but Runtime selection policy SHALL
remain authoritative over the final Kernel decision.

#### Scenario: Queue pressure

Given Scheduler reports backlog

When Kernel choice changes

Then decision is made through Runtime policy.