## ADDED Requirements

### Requirement: Autotuning Cache Is Separate From Artifact Cache

Autotuning Records SHALL remain logically separate from Kernel Artifact content.

#### Scenario: Kernel binary retained

Given tuning record is invalidated

When artifact cache is inspected

Then valid compiled binary need not be deleted.

---

### Requirement: Tuning Cache Uses Context Fingerprint

Autotuning cache entry SHALL identify workload/target/policy context.

#### Scenario: Different sequence bucket

Given tuning winner covers sequence 1..512

When sequence 8192 is requested

Then cache record is not assumed compatible.

---

### Requirement: Tuning Cache Hit Does Not Bypass Eligibility

Cached winner SHALL still pass current Kernel eligibility checks.

#### Scenario: Winner later revoked

Given cached tuning result names revoked Kernel

When selection runs

Then candidate is excluded.