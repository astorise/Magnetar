## ADDED Requirements

### Requirement: Kernel Exposes Selection Metadata

Any KernelAdvertisement metadata SHALL accurately describe the Kernel's actual behavior, and MAY expose policy-relevant metadata such as performance, workspace, determinism and specialization.

#### Scenario: High-workspace Kernel

Given Kernel needs 128 MiB workspace

When selection evaluates memory profile

Then workspace requirement participates in decision.

---

### Requirement: Performance Metadata Does Not Define Semantics

Kernel performance metadata SHALL NOT change Operator semantics.

#### Scenario: Faster approximate kernel

Given approximation changes numerical contract

When semantics do not allow approximation

Then Kernel is incompatible regardless of benchmark.

---

### Requirement: Runtime-Relevant Variant Differences Require Distinct Candidate

Provider variants differing in Runtime-relevant semantics or constraints SHALL
be separately represented.

#### Scenario: Deterministic and nondeterministic implementations

Given Provider has both variants

When determinism differs

Then they SHALL be distinguishable candidates rather than invisible private
switch.