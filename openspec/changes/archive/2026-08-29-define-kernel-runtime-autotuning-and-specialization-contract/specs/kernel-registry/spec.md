## ADDED Requirements

### Requirement: Registry Can Distinguish Specialization Instances

Kernel Registry SHALL retain enough identity to distinguish Runtime-relevant
specializations.

#### Scenario: Two Attention variants

Given two prepared variants have different tile specialization

When candidates are evaluated

Then each may carry distinct performance evidence.

---

### Requirement: Registry Does Not Tune Automatically

Kernel Registry SHALL not start benchmarks or compilation as a side effect of
candidate lookup.

#### Scenario: Candidate lookup

Given tuning record is absent

When Registry returns candidates

Then no autotuning session is implicitly started.

---

### Requirement: Registry Selection Uses Normal Policy After Tuning

Autotuning evidence MAY inform ranking but SHALL not bypass Registry/Runtime
eligibility.

#### Scenario: Winner revoked

Given tuning record names fastest candidate

But Registry marks it revoked

Then it is excluded.