## ADDED Requirements

### Requirement: Compilation Is Provider Capability

Kernel source compilation SHALL belong to Provider-level capability.

#### Scenario: Device discovered

Given Device metadata is available

When compilation capability is queried

Then capability belongs to Provider rather than Device object.

---

### Requirement: Provider May Support Execution Without Compilation

Provider execution capability SHALL not imply compilation capability.

#### Scenario: Static Provider

Given Provider contains built-in kernels only

When it registers

Then it remains valid without Kernel Compilation Capability.

---

### Requirement: Provider May Support Preparation Without Compilation

A Provider that prepares compatible AOT artifacts SHALL NOT be required to implement source compilation to do so.

#### Scenario: Precompiled binary

Given compatible Compiled Kernel Artifact exists

When Provider prepares it

Then source compiler is not required.

---

### Requirement: Compiler Native State Is Provider Private

Compiler process, compiler objects, driver compiler handles, and native Device handles SHALL remain Provider-private.

#### Scenario: Compilation observation

Given Provider invokes native compiler

When Runtime sees status

Then it sees opaque job metadata rather than native compiler handle.