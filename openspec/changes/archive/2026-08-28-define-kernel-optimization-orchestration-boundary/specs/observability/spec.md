## ADDED Requirements

### Requirement: Optimization Observability Is Distinguishable

Optimization Campaign observations SHALL be distinguishable from inference
execution observations.

#### Scenario: Candidate benchmark starts

Given benchmark begins

When observation is emitted

Then event is classified as optimization-plane activity.

---

### Requirement: Campaign Correlation

Optimization events SHALL support correlation by campaign/candidate/artifact.

#### Scenario: Candidate promoted later

Given campaign created candidate

When Runtime promotes it

Then promotion may retain campaign/evidence correlation metadata.

---

### Requirement: Optimization Data Is Redacted

Optimization observations SHALL redact sensitive source/inference information
according to policy.

#### Scenario: Workload profile emitted

Given profile was derived from production workloads

When observation is exported

Then raw prompts and raw user documents are absent.

---

### Requirement: Generator Credentials Are Redacted

Observability SHALL never expose generator or artifact-registry credentials.

#### Scenario: External service authentication fails

Given API key is present internally

When error is logged

Then key value is absent.