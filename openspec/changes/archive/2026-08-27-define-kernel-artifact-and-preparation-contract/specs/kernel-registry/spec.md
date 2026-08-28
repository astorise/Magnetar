## ADDED Requirements

### Requirement: Registry Tracks Prepared Kernel Readiness

A Kernel candidate without an associated ready PreparedKernel SHALL NOT be
treated as immediately dispatchable. Kernel Registry MAY associate compatible
Kernel candidates with PreparedKernel state.

#### Scenario: Kernel not prepared

Given compatible artifact exists but no PreparedKernel is ready

When dispatch selection runs

Then candidate is not treated as immediately executable.

---

### Requirement: Registry Does Not Own Native Handles

Kernel Registry SHALL NOT store or dereference native executable pointers.

#### Scenario: Prepared CUDA Kernel

Given Provider owns native CUDA function

When Registry stores candidate

Then it stores opaque PreparedKernelId only.

---

### Requirement: Registry Supports Multiple Prepared Generations

An older Prepared Kernel generation SHALL remain valid for in-flight requests
until no active reference remains. Kernel Registry MAY temporarily index
multiple Prepared Kernel generations for the same logical Kernel.

#### Scenario: Hot replacement

Given generation 18 replaces 17

When new request is dispatched

Then policy may choose 18 while in-flight request continues using 17.

---

### Requirement: Registry Validates Artifact Compatibility

Kernel Registry SHALL use artifact metadata as part of compatibility
selection where applicable.

#### Scenario: Architecture mismatch

Given compiled artifact targets incompatible architecture

When candidate selection occurs

Then candidate is excluded.

---

### Requirement: Registry Does Not Compile

Kernel Registry SHALL NOT perform source compilation.

#### Scenario: Missing compiled artifact

Given only source artifact exists

When Registry selects candidates

Then it reports preparation unavailable rather than invoking a compiler itself.