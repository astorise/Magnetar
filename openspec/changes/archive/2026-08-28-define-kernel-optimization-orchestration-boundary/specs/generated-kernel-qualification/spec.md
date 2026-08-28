## ADDED Requirements

### Requirement: Optimization Orchestration Composes Qualification

Optimization Campaign SHALL use existing Kernel Qualification semantics rather
than defining weaker correctness checks.

#### Scenario: Candidate compiles

Given candidate compiles

When campaign evaluates it

Then required qualification still occurs before production recommendation.

---

### Requirement: Campaign Evidence Identifies Qualification

Optimization evidence SHALL reference qualification profile, suite and result.

#### Scenario: Recommended Kernel

Given candidate is recommended

When evidence is inspected

Then exact qualification evidence is identifiable.

---

### Requirement: Qualification Failure Prevents Qualified Recommendation

Candidate that fails mandatory qualification SHALL NOT be recommended as
qualified production candidate.

#### Scenario: Differential mismatch

Given candidate fails numerical comparison

When campaign ranking completes

Then candidate cannot become qualified recommendation.