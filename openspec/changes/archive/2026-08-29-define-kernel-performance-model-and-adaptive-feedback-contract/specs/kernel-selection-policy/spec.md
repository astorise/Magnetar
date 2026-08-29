## ADDED Requirements

### Requirement: Online Performance Evidence May Influence Ranking

Kernel Selection Policy MAY consume sufficiently compatible online Performance Model evidence, and consumed evidence SHALL be context-compatible with the candidate being ranked.

#### Scenario: Online evidence supersedes old benchmark

Given policy is online-preferred after sufficient samples

When online evidence is current

Then it may rank eligible candidate differently from stale offline benchmark.

---

### Requirement: Online Evidence Cannot Override Eligibility

Performance observations SHALL remain secondary to hard constraints.

#### Scenario: Fast Kernel becomes untrusted

Given online performance remains excellent

When trust policy denies candidate

Then selection excludes it.

---

### Requirement: Performance Feedback Respects Hysteresis

Selection policy SHOULD avoid re-ranking active Kernel on minor statistical noise, and Runtime SHALL apply a configured hysteresis threshold before demoting or replacing the active Kernel.

#### Scenario: One-percent swing

Given hysteresis threshold is larger

When online ranking changes slightly

Then active Kernel may remain selected.

---

### Requirement: Performance Health May Affect Preference

Confirmed performance degradation MAY reduce preference of otherwise eligible candidate, and such preference adjustments SHALL NOT bypass hard eligibility constraints.

#### Scenario: Active generation regressed

Given another known-good eligible Kernel exists

When policy re-evaluates

Then degraded candidate may be demoted.