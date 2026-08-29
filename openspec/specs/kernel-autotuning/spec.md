# kernel-autotuning Specification

## Purpose
This specification defines Runtime Autotuning: a bounded, target-local mechanism for evaluating already authorized Kernel candidates and Specialization Instances against a finite, explicitly declared specialization domain. It defines the Kernel Specialization Template, Specialization Axis and Instance identity, qualification-coverage inheritance rules, the Autotuning Plan/Session/Record lifecycle, tuning cache and freshness semantics, resource budgets and inference-protection guarantees, and the boundary separating Runtime Autotuning from the Optimization Plane's arbitrary generative code search.
## Requirements
### Requirement: Bounded Runtime Autotuning

Runtime Autotuning SHALL operate on a finite or explicitly bounded
specialization/candidate domain.

#### Scenario: Unbounded parameter

Given specialization exposes arbitrary integer without maximum

When Runtime Autotuning validates the template

Then template is rejected as unbounded.

---

### Requirement: Autotuning Does Not Generate Arbitrary Source

Runtime Autotuning SHALL NOT arbitrarily rewrite Kernel source.

#### Scenario: Better candidate needed

Given current bounded candidates are slow

When tuning completes

Then Runtime may report no satisfactory specialization but does not invoke an
AI source generator.

---

### Requirement: Specialization Template

Tunable Kernel SHALL explicitly define the implementation parameters that may
vary.

#### Scenario: Triton tiling

Given Kernel permits BLOCK_M values 32 and 64

When template is parsed

Then only declared values may become Runtime specialization instances.

---

### Requirement: Deterministic Specialization Identity

Specialization Instance SHALL have stable identity derived from template and
assignments.

#### Scenario: Assignment ordering differs

Given equivalent values are supplied in different metadata order

When identity is computed

Then Specialization Instance identity is identical.

---

### Requirement: Specialization Preserves Semantics

Tuning parameter SHALL not alter portable Operator semantics.

#### Scenario: Parameter changes output contract

Given proposed parameter changes MatMul semantics

When template is validated

Then it cannot be treated as ordinary specialization axis.

---

### Requirement: Explicit Qualification Coverage

Qualification evidence SHALL explicitly state which Specialization Instances or
envelope it covers.

#### Scenario: One variant qualified

Given BLOCK_M=32 is qualified

When BLOCK_M=64 is considered

Then Runtime does not assume qualification unless evidence explicitly covers it.

---

### Requirement: Tuning Is Not Qualification

Benchmark success SHALL not substitute for correctness qualification.

#### Scenario: Fast unqualified specialization

Given variant wins tuning benchmark but lacks required qualification

When production selection runs

Then variant remains ineligible.

---

### Requirement: No Decode Hot-Path Autotuning

Normal token decode SHALL not synchronously launch Runtime Autotuning Session.

#### Scenario: Tuning cache miss

Given no tuning record exists

When decode executes

Then known-good/default selection is used according to policy rather than
blocking for benchmark.

---

### Requirement: Workload-Aware Tuning

Autotuning Record SHALL identify workload context to which result applies.

#### Scenario: Decode result reused for prefill

Given winner was measured for decode workload only

When prefill executes

Then record is not automatically authoritative for prefill.

---

### Requirement: Autotuning Resource Budget

Autotuning SHALL have bounded resource policy.

#### Scenario: Candidate budget exceeded

Given plan permits 16 candidate evaluations

When 16 have run

Then additional candidate is not benchmarked without new policy.

---

### Requirement: Known-Good Kernel Preserved

Autotuning failure SHALL not remove current known-good Kernel.

#### Scenario: All specializations fail

Given current Kernel is active

When tuning session finds no valid candidate

Then current Kernel remains active.

---

### Requirement: Tuning Winner Is Advisory To Selection

Autotuning winner SHALL pass normal Kernel Selection and promotion policy.

#### Scenario: Fast winner becomes memory-infeasible

Given memory pressure changes after tuning

When selection runs

Then winner may be rejected.

---

### Requirement: Tuning Cache Is Context Sensitive

Cached tuning result SHALL include target/workload/policy compatibility.

#### Scenario: Driver update

Given cached result was produced under incompatible previous driver

When lookup occurs

Then record is stale or incompatible.

---

### Requirement: Reproducible Mode

Runtime SHALL support disabling live autotuning for reproducible execution.

#### Scenario: Pinned Model Instance

Given Model Instance pins tuning result

When new faster variant becomes available

Then live tuning does not silently change its Kernel choice.

---

### Requirement: Adaptive Feedback May Request Re-Tuning

Kernel Performance Model MAY request bounded Runtime Autotuning, and any such request SHALL remain within existing Autotuning template and candidate boundaries.

#### Scenario: Tuning evidence becomes stale

Given performance drift is confirmed

When policy permits

Then a new Autotuning Session may be scheduled for affected workload bucket.

---

### Requirement: Re-Tuning Uses Existing Boundaries

Adaptive re-tuning SHALL obey normal Autotuning template, candidate,
qualification, resource and hot-path restrictions.

#### Scenario: Performance signal requests arbitrary compiler flags

Given requested change is outside declared specialization domain

When re-tuning plan validates

Then request is rejected/escalated externally.

---

### Requirement: Re-Tuning Is Rate Limited

Repeated feedback SHALL not create continuous autotuning loops.

#### Scenario: Noisy environment

Given repeated drift signals occur inside cooldown

When new requests are evaluated

Then redundant re-tuning is suppressed.
