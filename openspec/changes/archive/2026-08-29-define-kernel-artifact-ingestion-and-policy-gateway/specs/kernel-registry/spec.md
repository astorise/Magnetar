## ADDED Requirements

### Requirement: Registry Discovers Only Accepted Candidates

Normal Kernel Registry candidate discovery SHALL exclude staged, quarantined,
rejected, and revoked ingestion artifacts.

#### Scenario: Candidate still validating

Given ingestion transaction has not committed

When Registry discovers Operator implementations

Then candidate is absent.

---

### Requirement: Registry Publication Follows Commit

Candidate metadata SHALL NOT become Registry-discoverable until ingestion
commit succeeds and any additional required registration policy is satisfied.

#### Scenario: Commit succeeds

Given accepted Kernel is committed

When candidate registration runs

Then Registry may index it according to normal policy.

---

### Requirement: Ingestion Failure Does Not Mutate Registry

Failed import SHALL not partially create Registry candidate.

#### Scenario: Qualification evidence malformed

Given manifest parsing succeeded but evidence validation failed

When transaction rejects

Then Registry state is unchanged.