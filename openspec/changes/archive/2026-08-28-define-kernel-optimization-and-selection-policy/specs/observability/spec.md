## ADDED Requirements

### Requirement: Kernel Selection Observability

Recorded Kernel selection lifecycle events SHALL exclude native handles and raw tensor data; Runtime SHOULD record this redacted lifecycle.

#### Scenario: Candidate selected

Given five candidates are evaluated

When selection completes

Then selected Kernel and policy profile may be observed.

---

### Requirement: Candidate Exclusion Is Observable

Exposed exclusion reasons SHALL identify the specific constraint that excluded the candidate; Runtime SHOULD expose such structured reasons.

#### Scenario: Candidate excluded by affinity

Given candidate fails Resource Affinity

When diagnostics are requested

Then affinity incompatibility is reported.

---

### Requirement: Ranking Metadata Is Redacted

Selection observations SHALL not expose native handles or model data.

#### Scenario: Ranking report

Given candidates are ranked

When report is exported

Then it may contain score, KernelId and artifact digest but no tensor values or
native addresses.

---

### Requirement: Hysteresis Is Observable

The record of a hysteresis-retained active Kernel SHALL include the comparison that was suppressed; Runtime SHOULD record such retention events.

#### Scenario: Marginal improvement

Given new candidate is slightly faster but below promotion threshold

When active Kernel remains

Then decision reason is observable.

---

### Requirement: Fallback Is Observable

Explicit fallback SHALL record original failure class and selected fallback.

#### Scenario: GPU unavailable

Given Runtime falls back to Reference CPU

When observation is emitted

Then fallback policy and reason are included.