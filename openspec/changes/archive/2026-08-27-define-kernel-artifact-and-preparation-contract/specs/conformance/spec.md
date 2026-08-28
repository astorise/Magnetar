## ADDED Requirements

### Requirement: Kernel Artifact Conformance

Conformance SHALL validate Kernel Source Artifact, Compiled Kernel Artifact,
and Prepared Kernel lifecycle separation.

#### Scenario: Lifecycle test

Given source artifact is compiled and prepared

When conformance runs

Then each stage has distinct identity and ownership.

---

### Requirement: Device Compilation Boundary Conformance

Conformance SHALL validate Device does not perform compilation.

#### Scenario: Device API audit

Given Device public contract is inspected

When conformance runs

Then arbitrary source compilation capability is absent.

---

### Requirement: Scheduler Compilation Boundary Conformance

Conformance SHALL validate Scheduler does not compile kernels.

#### Scenario: Scheduler API audit

Given Scheduler is inspected

When conformance runs

Then compiler ownership is absent.

---

### Requirement: Native Handle Conformance

Conformance SHALL validate native kernel handles remain Provider-private.

#### Scenario: Public API audit

Given Runtime, WIT, Registry, Device, and diagnostics are inspected

When conformance runs

Then no native kernel pointer or Provider executable handle is exposed.

---

### Requirement: Hot Path Compilation Conformance

Conformance SHALL validate normal decode path does not perform synchronous
kernel compilation.

#### Scenario: Unprepared kernel during decode

Given decode requires unprepared kernel

When conformance runs

Then structured readiness/admission error occurs rather than compilation.

---

### Requirement: Artifact Trust Conformance

Conformance SHALL validate artifact origin, format, AI provenance, local
location, and cache presence do not imply trust.

#### Scenario: AI-generated cached artifact

Given artifact is AI-generated and cached

When trust policy has not approved it

Then it remains untrusted.

---

### Requirement: Operator Semantics Conformance

Conformance SHALL validate Kernel Artifact semantics against portable Operator
semantics.

#### Scenario: Invalid generated MatMul

Given generated Kernel does not preserve MatMul semantics

When qualification/conformance evaluates it

Then it cannot become an eligible Kernel.

---

### Requirement: Prepared Generation Coexistence Conformance

Conformance SHALL validate multiple Prepared Kernel generations can coexist
without destroying in-flight kernel state.

#### Scenario: Replacement during execution

Given generation 1 has active invocation

When generation 2 becomes current

Then generation 1 remains valid until active references reach zero.