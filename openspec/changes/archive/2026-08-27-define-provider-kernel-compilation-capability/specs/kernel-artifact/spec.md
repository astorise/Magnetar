## ADDED Requirements

### Requirement: Compilation Produces Compiled Kernel Artifact

Provider compilation SHALL produce a Compiled Kernel Artifact rather than a
Prepared Kernel directly at the logical contract level.

#### Scenario: Triton compiled

Given Triton source compiles successfully

When result is accepted

Then a Compiled Kernel Artifact exists before Provider preparation.

---

### Requirement: Compiled Artifact Records Compiler Identity

Compiled Kernel Artifact SHALL record compiler identity and version where available.

#### Scenario: Compiler upgrade

Given same source is compiled with different compiler version

When artifact metadata is compared

Then compiler identity difference is observable.

---

### Requirement: Compiler Options Affect Artifact Identity

Compiler settings affecting output SHALL participate in artifact identity or
compatibility fingerprint.

#### Scenario: Fast-math changed

Given compiler changes fast-math setting

When new artifact is produced

Then it is not treated as indistinguishable from previous artifact.

---

### Requirement: Compiled Artifact Records Target

Compiled Kernel Artifact SHALL identify target compatibility.

#### Scenario: sm90 artifact

Given binary was compiled for sm90

When Device is incompatible

Then preparation or Registry selection rejects it.

---

### Requirement: Compilation Output Has Digest

Compiled Kernel Artifact SHALL have integrity digest before readiness.

#### Scenario: Output mutated

Given compiled bytes change after digest calculation

When validation runs

Then integrity failure is returned.

---

### Requirement: Compilation Does Not Qualify Artifact

Compilation success SHALL remain distinct from semantic qualification.

#### Scenario: Numerically incorrect kernel

Given source compiles but produces wrong results

When compilation completes

Then artifact is not automatically production-eligible.