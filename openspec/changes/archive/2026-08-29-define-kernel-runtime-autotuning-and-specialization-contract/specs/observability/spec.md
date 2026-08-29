## ADDED Requirements

### Requirement: Autotuning Lifecycle Is Observable

Runtime MAY emit redacted observations for tuning planning, preparation, measurement and result, and emitted observations SHALL be redacted per Autotuning Payloads Are Redacted.

#### Scenario: Winner selected

Given tuning session completes

When observations are exported

Then winning specialization, workload bucket and policy may be reported.

---

### Requirement: Tuning Versus Production Execution Is Distinguishable

Observability SHALL distinguish benchmark/tuning execution from inference
execution.

#### Scenario: Kernel invocation during benchmark

Given candidate runs as tuning fixture

When observed

Then invocation is classified as autotuning activity.

---

### Requirement: Autotuning Payloads Are Redacted

Observability SHALL not expose raw tuning tensors or native handles by default.

#### Scenario: Candidate benchmark fails

Given diagnostic exists

When exported

Then stable IDs and failure category may appear while raw fixture data does not.