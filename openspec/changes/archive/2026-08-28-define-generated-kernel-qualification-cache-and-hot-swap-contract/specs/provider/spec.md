## ADDED Requirements

### Requirement: Provider Supports Prepared Kernel Generations

Provider SHALL allow Runtime to distinguish Prepared Kernel generations without
exposing native handles.

#### Scenario: Kernel replacement

Given Provider prepares replacement kernel

When returned

Then it receives distinct opaque PreparedKernelId/generation state.

---

### Requirement: Provider Safe Kernel Destruction

Provider SHALL not destroy native Prepared Kernel state while active work still
uses it according to Runtime/Provider lifetime protocol.

#### Scenario: In-flight GPU kernel

Given old generation has active invocation

When Runtime retires it

Then destruction waits for safe state.

---

### Requirement: Provider Remains Loaded During Kernel Hot Swap

Kernel hot swap SHALL NOT require Provider unload.

#### Scenario: New CUBIN promoted

Given CUDA Provider has active context and streams

When replacement kernel is promoted

Then Provider stays loaded.

---

### Requirement: Provider Failure Does Not Corrupt Active Candidate

Failure preparing or destroying one Kernel SHALL not silently invalidate
unrelated active Prepared Kernels.

#### Scenario: v2 preparation fails

Given v1 is active

When Provider fails to prepare v2

Then v1 remains valid.