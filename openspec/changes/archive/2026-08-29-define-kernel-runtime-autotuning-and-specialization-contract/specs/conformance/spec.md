## ADDED Requirements

### Requirement: Bounded Autotuning Conformance

Conformance SHALL prove Runtime Autotuning cannot evaluate an unbounded
specialization space.

#### Scenario: Unbounded template

Given tuning axis lacks explicit bound

When plan validates

Then it is rejected.

---

### Requirement: No Arbitrary Generation Conformance

Conformance SHALL prove Runtime Autotuning cannot invoke arbitrary Kernel source
generation.

#### Scenario: Candidate set exhausted

Given no candidate meets objective

When tuning ends

Then no external AI generator is invoked by Runtime.

---

### Requirement: No Arbitrary Compiler Flag Conformance

Conformance SHALL reject arbitrary free-form compiler arguments as tuning axes.

#### Scenario: Manifest exposes arbitrary flags

Given specialization contains unrestricted compiler command string

When validated

Then template is rejected.

---

### Requirement: No Hot-Path Tuning Conformance

Conformance SHALL prove token decode does not block on autotuning.

#### Scenario: Missing tuning cache

Given tuning record absent

When token generated

Then benchmark is not synchronously launched.

---

### Requirement: Accepted Artifact Requirement Conformance

Conformance SHALL prove quarantined/rejected artifacts cannot participate in
Runtime Autotuning.

#### Scenario: Quarantined Kernel

Given specialization template exists

When tuning candidates enumerate

Then Kernel is absent.

---

### Requirement: Qualification Coverage Conformance

Conformance SHALL prove specialization uses only appropriate qualification
evidence.

#### Scenario: Qualified exact instance differs

Given variant A is qualified and variant B is not covered

When tuning ranks both

Then B cannot become production-eligible solely from benchmark.

---

### Requirement: Tuning Cache Context Conformance

Conformance SHALL prove incompatible workload/target context invalidates tuning
reuse.

#### Scenario: Different GPU architecture

Given record came from sm90

When incompatible target uses cache

Then record is rejected/stale.

---

### Requirement: Memory Authority Conformance

Conformance SHALL prove Memory Manager may reject a tuning candidate regardless
of benchmark potential.

#### Scenario: Workspace infeasible

Given candidate would be fastest

When workspace fails admission

Then it is not benchmarked/selected as production candidate.

---

### Requirement: Known-Good Preservation Conformance

Conformance SHALL prove tuning failure cannot remove active known-good Kernel.

#### Scenario: Every candidate crashes during benchmark

Given current Kernel is healthy

When tuning fails

Then current Kernel remains active.

---

### Requirement: Tuning Winner Selection Boundary Conformance

Conformance SHALL prove tuning winner cannot bypass Kernel Selection Policy.

#### Scenario: Winner later untrusted

Given tuning identifies fastest variant

When trust policy rejects it

Then Runtime does not execute it.

---

### Requirement: Reproducible Mode Conformance

Conformance SHALL prove reproducible Model Instance cannot silently change
specialization through live tuning.

#### Scenario: Faster specialization discovered

Given Model Instance is pinned

When background tuning produces new winner

Then pinned instance remains unchanged.

---

### Requirement: Prepared State Persistence Conformance

Conformance SHALL prove Autotuning Record does not persist native
PreparedKernelId as portable tuning identity.

#### Scenario: Runtime restart

Given cached tuning record exists

When Runtime restarts

Then required Kernel is prepared again and native handle is not restored from
record.