## ADDED Requirements

### Requirement: Provider May Expose Execution Timing

Provider MAY expose timing capability suitable for Runtime Performance Model, and when exposed, timing observations SHALL be tagged with their measurement method.

#### Scenario: GPU event timing available

Given Provider can measure Device execution

When observation is created

Then timing method is identified as Provider/Device event timing.

---

### Requirement: Timing Capability Is Optional

Provider SHALL remain valid if precise Device timing is unavailable.

#### Scenario: Reference CPU Provider

Given only host timing is available

When performance observations are collected

Then host timing may be used with corresponding method metadata.

---

### Requirement: Provider Metrics Do Not Override Runtime Policy

Provider performance metrics SHALL be evidence, not selection authority.

#### Scenario: Provider claims Kernel fastest

Given Runtime evidence/policy disagrees

When selection occurs

Then Runtime remains authoritative.