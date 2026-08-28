## ADDED Requirements

### Requirement: Qualification Lifecycle Observability

Observability SHALL NOT expose raw kernel source, compiled binaries, or test tensors by default.

Runtime SHOULD emit redacted qualification lifecycle observations.

#### Scenario: Differential mismatch

Given candidate output exceeds tolerance

When qualification fails

Then observation identifies Kernel/artifact/profile and mismatch category
without dumping raw tensor data by default.

---

### Requirement: Benchmark Observability

A benchmark summary without workload/context metadata SHALL NOT be treated as authoritative ranking evidence.

Runtime MAY emit benchmark summaries without exposing sensitive inputs.

#### Scenario: Candidate benchmark

Given benchmark completes

When observation is emitted

Then latency/throughput/profile metadata may be included.

---

### Requirement: Promotion Observability

Kernel promotion SHALL be observable.

#### Scenario: Generation 42 promoted

Given candidate becomes active

When promotion completes

Then event records logical Kernel, old generation, new generation and policy
decision.

---

### Requirement: Rollback Observability

Rollback SHALL be observable.

#### Scenario: Regression rollback

Given active kernel is rolled back

When event is emitted

Then reason and resulting active generation are recorded.

---

### Requirement: Revocation Observability

Revocation SHALL be observable without exposing executable internals.

#### Scenario: Kernel revoked

Given qualification is revoked

When event is emitted

Then artifact digest and reason may be reported without native handles.

---

### Requirement: Generated Kernel Observability Redaction

Observability SHALL redact raw source, compiled binaries, raw qualification
tensors and native handles by default.

#### Scenario: Qualification crash

Given Provider fails during generated kernel test

When diagnostic is emitted

Then sensitive artifact/tensor contents are absent by default.