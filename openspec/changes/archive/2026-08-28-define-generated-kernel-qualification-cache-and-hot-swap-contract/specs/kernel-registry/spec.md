## ADDED Requirements

### Requirement: Registry Considers Qualification Eligibility

Kernel Registry SHALL consider qualification status when policy requires it.

#### Scenario: Faster unqualified Kernel

Given unqualified candidate benchmarks faster

When production selection runs

Then it cannot outrank eligible qualified candidates.

---

### Requirement: Registry Promotion Is Explicit

Candidate Kernel SHALL become active only through explicit promotion.

#### Scenario: Candidate prepared

Given candidate is prepared successfully

When no promotion occurs

Then it does not automatically become active.

---

### Requirement: Atomic Kernel Promotion

Dispatch SHALL NOT observe a partially updated Registry state.

Registry promotion SHOULD be atomic from dispatch perspective.

#### Scenario: Promotion races with dispatch

Given promotion occurs concurrently with new invocation

When dispatch resolves Kernel

Then invocation observes complete old or complete new Registry generation.

---

### Requirement: Multiple Prepared Generations

Each tracked generation SHALL be uniquely identified.

Registry MAY track multiple Prepared Kernel generations.

#### Scenario: Hot swap

Given generation 2 is promoted

When generation 1 has in-flight work

Then both generations may coexist temporarily.

---

### Requirement: Retiring Generation Receives No New Work

After Kernel generation enters retiring state, Registry SHALL stop selecting it
for new work.

#### Scenario: New request after promotion

Given old Kernel is retiring

When request resolves

Then new active generation is selected where compatible.

---

### Requirement: Revoked Kernel Not Selected

Registry SHALL not select revoked Kernel for new work.

#### Scenario: Security revocation

Given active Kernel is revoked

When next dispatch occurs

Then revoked Kernel is not selected.

---

### Requirement: Performance Ranking Follows Eligibility

Performance ranking SHALL occur after compatibility, qualification, trust and
policy eligibility.

#### Scenario: Incorrect fastest candidate

Given incorrect candidate has best benchmark

When Registry ranks

Then candidate remains ineligible.