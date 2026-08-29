## ADDED Requirements
### Requirement: Selection Result May Be Materialized Into Plan

Kernel Selection Policy SHALL be able to materialize its decision into Prepared Execution
Plan binding.

#### Scenario: Latency profile selects candidate B

Given candidate B passes hard filters and ranks first

When Plan is built

Then B becomes explicit binding.

---

### Requirement: Plan Prevents Per-Invocation Selection Churn

Selection policy SHALL be able to retain a Plan binding until staleness/replan policy
requires reconsideration.

#### Scenario: Tiny pressure variation

Given current Plan remains valid

When Device pressure changes slightly

Then Runtime need not rerun global selection for every operation.

---

### Requirement: Hard Policy Change Invalidates Plan

A new policy that makes existing binding ineligible SHALL invalidate dependent
Plan.

#### Scenario: Determinism becomes mandatory

Given active Plan uses nondeterministic Kernel

When hard deployment policy changes

Then Plan cannot receive new work.

---

### Requirement: Preference Change May Mark Plan Stale

A non-hard ranking preference change SHALL be able to mark Plan stale rather than invalid.

#### Scenario: Throughput becomes preferred over latency

Given current Kernel remains eligible

When profile changes

Then replacement may be built while old Plan remains usable according to policy.
