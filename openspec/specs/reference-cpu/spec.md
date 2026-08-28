# reference-cpu Specification

## Purpose
This specification defines the Reference CPU Provider's role as the default correctness oracle for generated Kernel qualification: recording oracle identity and version in qualification metadata, prioritizing semantic correctness over matching generated-kernel performance techniques, and excluding Reference CPU performance from production performance baselines.
## Requirements
### Requirement: Reference CPU May Serve As Qualification Oracle

When used as oracle, Reference CPU identity and version SHALL be recorded in qualification metadata.

Reference CPU Provider SHOULD serve as correctness oracle for supported portable
Operators.

#### Scenario: GPU candidate qualification

Given generated GPU Softmax is tested

When oracle comparison runs

Then Reference CPU Softmax provides baseline semantics.

---

### Requirement: Reference CPU Oracle Prioritizes Correctness

Reference CPU oracle SHALL not change semantics merely to match generated
Kernel performance techniques.

#### Scenario: Generated approximation

Given candidate uses approximate math

When compared to Reference CPU

Then approximation is accepted only through explicit tolerance policy.

---

### Requirement: Reference Oracle Version Is Recorded

Qualification evidence SHALL record reference implementation identity/version.

#### Scenario: Reference implementation fixed

Given Reference CPU behavior changes due to bug fix

When old qualification evidence is inspected

Then old oracle version remains identifiable.

---

### Requirement: Reference CPU Is Not Performance Oracle

Reference CPU performance SHALL NOT be required as production performance
baseline.

#### Scenario: Generated GPU kernel

Given GPU kernel is much faster

When ranking is performed

Then Reference CPU contributes correctness evidence, not comparable GPU
performance expectation.

