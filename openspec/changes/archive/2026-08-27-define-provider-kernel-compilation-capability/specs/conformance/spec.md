## ADDED Requirements

### Requirement: Provider Compilation Capability Conformance

Conformance SHALL validate optional Provider Kernel Compilation Capability.

#### Scenario: Provider without compiler

Given Provider has no compilation capability

When conformance core profile runs

Then Provider can still pass non-compilation conformance.

---

### Requirement: Source Format Negotiation Conformance

Conformance SHALL validate unsupported source formats are rejected before
compiler invocation.

#### Scenario: WGSL to CPU Provider

Given Provider does not accept WGSL

When compile is requested

Then structured unsupported format error is returned.

---

### Requirement: Compilation Job Lifecycle Conformance

Conformance SHALL validate compilation job state transitions.

#### Scenario: Successful async job

Given compilation is asynchronous

When polled

Then states progress legally to succeeded and cannot revert to compiling.

---

### Requirement: Compilation Cancellation Conformance

Conformance SHALL validate declared cancellation behavior.

#### Scenario: Cancelled compilation

Given Provider declares cooperative cancellation

When cancel is requested

Then job does not publish valid partial output.

---

### Requirement: Compilation Deadline Conformance

Conformance SHALL validate declared deadline behavior.

#### Scenario: Compiler exceeds deadline

Given deadline is enforceable

When compiler exceeds it

Then job ends timed-out without ready artifact.

---

### Requirement: Compilation Isolation Conformance

Conformance SHALL validate Runtime policy can reject insufficient isolation.

#### Scenario: Untrusted source

Given policy requires sandboxed compilation

When Provider advertises in-process compiler only

Then compilation is denied.

---

### Requirement: Compilation Trust Separation Conformance

Conformance SHALL validate compilation success does not imply trust or
qualification.

#### Scenario: Compilable untrusted source

Given untrusted source compiles

When output is created

Then output remains untrusted/unqualified according to policy.

---

### Requirement: Provider Kernel Compilation Hot Path Conformance

Conformance SHALL validate Kernel execution cannot silently invoke compiler.

#### Scenario: Missing Prepared Kernel

Given execution begins without PreparedKernel

When dispatch occurs

Then structured failure happens instead of compilation.

---

### Requirement: ABI Ownership Conformance

Conformance SHALL validate all compilation ABI buffers use declared ownership
and release paths.

#### Scenario: Result buffer

Given Provider allocates result buffer

When Runtime consumes result

Then required release callback is invoked exactly according to contract.

---

### Requirement: ABI Handle Opacity Conformance

Conformance SHALL validate job IDs and PreparedKernelIds are opaque.

#### Scenario: Numeric handle

Given handle is represented as integer

When public API/diagnostics inspect it

Then no native pointer semantics are exposed.

---

### Requirement: Compiler Failure Atomicity Conformance

Conformance SHALL validate compiler failure leaves existing known-good Kernel
state intact.

#### Scenario: Replacement compile fails

Given Kernel v1 is prepared

And compilation of v2 crashes

When job fails

Then v1 remains usable and v2 is not published.