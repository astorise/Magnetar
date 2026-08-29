## ADDED Requirements

### Requirement: Performance Evidence Cannot Grant Trust

Conformance SHALL prove fast Kernel remains untrusted if trust policy denies it.

#### Scenario: Untrusted fastest Kernel

Given production observations are excellent

When selection runs

Then trust denial remains authoritative.

---

### Requirement: Performance Evidence Cannot Grant Qualification

Conformance SHALL prove observations do not substitute for qualification.

#### Scenario: Unqualified specialization performs correctly in sampled runs

Given no valid qualification evidence exists

When Performance Model updates

Then candidate remains unqualified.

---

### Requirement: Performance Context Isolation

Conformance SHALL prove observations do not leak across incompatible
artifact/specialization contexts.

#### Scenario: New binary

Given artifact digest changes

When new generation executes

Then historical metrics are not automatically attributed to it.

---

### Requirement: Sample Sufficiency Conformance

Conformance SHALL prove isolated outlier cannot trigger confirmed regression
when policy requires more evidence.

#### Scenario: One slow sample

Given minimum sample count is 100

When one sample is slow

Then regression remains unconfirmed.

---

### Requirement: Drift Detection Conformance

Conformance SHALL detect sustained compatible difference from benchmark
baseline.

#### Scenario: Online latency increases materially

Given enough samples exceed threshold

When model updates

Then drift state is produced.

---

### Requirement: Workload Drift Conformance

Conformance SHALL identify substantial workload bucket distribution change.

#### Scenario: Production moves to larger batches

Given original tuning workload differs

When distribution crosses policy threshold

Then workload-drift event is emitted.

---

### Requirement: Bounded Re-Tuning Conformance

Conformance SHALL prove feedback-triggered tuning remains inside declared
specialization domain.

#### Scenario: Better code needed

Given no bounded variant meets goal

When retuning ends

Then Runtime does not invent source.

---

### Requirement: No Hot-Path Adaptive Benchmarking

Conformance SHALL prove active decode does not benchmark alternatives
synchronously.

#### Scenario: Regression during decode

Given regression is detected

When token loop continues

Then background retuning/fallback occurs according to policy.

---

### Requirement: External Escalation Boundary

Conformance SHALL prove unresolved Runtime tuning produces external optimization
signal, not direct AI generation.

#### Scenario: All specializations poor

Given policy permits external optimization

When Runtime escalates

Then no generator is invoked inside Runtime.

---

### Requirement: Hysteresis Conformance

Conformance SHALL prove small measurement noise does not create repeated Kernel
switching.

#### Scenario: Ranking alternates slightly

Given difference stays below threshold

When observations update

Then active Kernel remains stable.

---

### Requirement: Reproducible Mode Conformance

Conformance SHALL prove online evidence does not alter pinned Kernel choice.

#### Scenario: Another Kernel becomes faster

Given instance is pinned

When model recommends alternate Kernel

Then selection remains pinned.

---

### Requirement: Bounded Retention Conformance

Conformance SHALL prove continuous inference does not produce unbounded
performance telemetry memory growth.

#### Scenario: Million observations

Given retention limits exist

When traffic continues

Then old raw samples are aggregated/expired.

---

### Requirement: Feedback Failure Isolation

Conformance SHALL prove failure of Performance Model does not corrupt active
known-good Kernel.

#### Scenario: Aggregator fails

Given active Kernel remains healthy

When feedback subsystem errors

Then active execution remains valid.

---

### Requirement: Telemetry Redaction Conformance

Conformance SHALL prove exported performance evidence contains no raw prompt,
weight, KV, native handle, secret or credential.

#### Scenario: Export generated

Given aggregate telemetry exists

When inspected

Then sensitive inference content is absent.