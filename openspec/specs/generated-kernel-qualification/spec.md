# generated-kernel-qualification Specification

## Purpose
This specification defines the qualification layer for generated Kernel implementations: differential correctness against a reference oracle, explicit tolerance and shape/compatibility envelopes, qualification status and profiles, the separation of qualification from trust, performance benchmarking and ranking order, and failure atomicity guarantees that keep a failed candidate from affecting the currently active Kernel.
## Requirements
### Requirement: Generated Kernel Qualification

Generated Kernel SHALL require explicit qualification before becoming
production eligible when policy requires qualification.

#### Scenario: Compiled generated kernel

Given generated kernel compiles successfully

When production eligibility is evaluated

Then compilation success alone is insufficient.

---

### Requirement: Qualification Status

Qualification SHALL expose explicit status.

#### Scenario: Qualification fails

Given numerical comparison fails

When qualification completes

Then Kernel becomes rejected rather than qualified.

---

### Requirement: Qualification Profile

Qualification SHALL identify the profile and version against which evidence was
produced.

#### Scenario: Baseline versus strict

Given Kernel passed baseline correctness

When strict qualification is requested

Then baseline evidence is not silently treated as strict evidence.

---

### Requirement: Reference Correctness Oracle

When Reference CPU is used as oracle, its identity and version SHALL be recorded in qualification metadata.

Qualification SHOULD use Reference CPU Provider as correctness oracle for
portable baseline Operators where supported.

#### Scenario: Generated MatMul

Given generated MatMul targets GPU

When baseline qualification runs

Then output may be compared with Reference CPU MatMul.

---

### Requirement: Differential Correctness

Qualification SHALL reject output outside declared correctness/tolerance rules.

#### Scenario: Wrong result

Given generated output differs from oracle beyond tolerance

When qualification runs

Then Kernel is rejected.

---

### Requirement: Explicit Numerical Tolerance

Numerical tolerance SHALL be explicit where exact equality is not required.

#### Scenario: FP16 kernel

Given generated FP16 kernel has approximate accumulation

When compared

Then declared tolerance profile controls acceptance.

---

### Requirement: Qualification Matrix

Qualification SHALL evaluate more than a single happy-path input according to
profile.

#### Scenario: Shape-specialized attention

Given kernel advertises range of sequence lengths

When qualification runs

Then representative boundary/irregular shapes are evaluated.

---

### Requirement: Qualification Envelope

Qualification evidence SHALL identify compatibility envelope.

#### Scenario: Limited evidence

Given Kernel was qualified only to sequence length 4096

When sequence length 8192 is requested

Then qualification evidence does not automatically cover that execution.

---

### Requirement: Fused Kernel Equivalence

Fused Kernel SHALL preserve semantics of the Operator group it replaces.

#### Scenario: Fused RMSNorm+MatMul

Given fused kernel is qualified

When reference is evaluated

Then result is compared to unfused reference composition.

---

### Requirement: Determinism Qualification

Kernel advertising deterministic behavior SHALL pass determinism qualification
according to policy.

#### Scenario: Repeated execution differs

Given identical input produces differing results beyond allowed rules

When deterministic profile is used

Then qualification fails.

---

### Requirement: Trust And Qualification Are Independent

Trust and qualification SHALL remain separate dimensions.

#### Scenario: Trusted vendor binary

Given artifact is trusted but has no qualification evidence

When production policy requires qualification

Then it is not production eligible.

---

### Requirement: Qualification Failure Atomicity

Failed candidate qualification SHALL NOT affect current active Kernel.

#### Scenario: Generated v2 incorrect

Given v1 is active

When v2 qualification fails

Then v1 remains active.

---

### Requirement: Optimization Orchestration Composes Qualification

Optimization Campaign SHALL use existing Kernel Qualification semantics rather
than defining weaker correctness checks.

#### Scenario: Candidate compiles

Given candidate compiles

When campaign evaluates it

Then required qualification still occurs before production recommendation.

---

### Requirement: Campaign Evidence Identifies Qualification

Optimization evidence SHALL reference qualification profile, suite and result.

#### Scenario: Recommended Kernel

Given candidate is recommended

When evidence is inspected

Then exact qualification evidence is identifiable.

---

### Requirement: Qualification Failure Prevents Qualified Recommendation

Candidate that fails mandatory qualification SHALL NOT be recommended as
qualified production candidate.

#### Scenario: Differential mismatch

Given candidate fails numerical comparison

When campaign ranking completes

Then candidate cannot become qualified recommendation.

