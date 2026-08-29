## ADDED Requirements

### Requirement: Adaptive Performance Observability

Runtime MAY expose redacted aggregate Kernel performance state, and exposed state SHALL be redacted of raw model and user content.

#### Scenario: Kernel regressed

Given sufficient evidence confirms regression

When observability is queried

Then Kernel/workload bucket/regression reason may be reported.

---

### Requirement: Online And Offline Evidence Distinguished

Observability SHOULD identify evidence source. Observability output SHALL
label each metric with its evidence source (online or offline).

#### Scenario: Benchmark and production differ

Given both values exist

When diagnostics are emitted

Then offline benchmark and online observation are not conflated.

---

### Requirement: Re-Tuning Request Is Observable

Adaptive re-tuning request SHOULD be observable, and when emitted, Runtime SHALL include re-tuning reason and target workload bucket in the observable record.

#### Scenario: Drift triggers tuning

Given workload changes materially

When re-tuning is requested

Then reason, bucket and policy version may be logged.

---

### Requirement: External Optimization Escalation Is Observable

Runtime SHOULD report when bounded Runtime adaptation cannot solve a performance problem, and when escalation occurs, Runtime SHALL record it distinctly from routine Runtime Autotuning events.

#### Scenario: No specialization meets SLO

Given external optimization request is emitted

When observation is recorded

Then escalation is distinguishable from Runtime autotuning.

---

### Requirement: Performance Telemetry Redaction

Performance observability SHALL exclude raw prompts, model weights, KV contents,
native handles and secrets by default.

#### Scenario: Telemetry export

Given metrics are exported

When payload is inspected

Then only aggregate workload/performance metadata is present.