## ADDED Requirements

### Requirement: Kernel Cache Is Not Tensor Residency

Persistent Kernel Artifact cache SHALL remain distinct from Runtime Tensor
residency.

#### Scenario: Cached CUBIN

Given CUBIN is stored on disk

When Memory Manager reports model tensor residency

Then CUBIN cache storage is not counted as resident inference tensor.

---

### Requirement: Prepared Kernel Executable Memory Is Distinct

Prepared Kernel executable memory SHALL remain logically distinct from Runtime
Tensor Resource memory.

#### Scenario: GPU module loaded

Given Provider allocates module memory

When tensor memory accounting runs

Then executable module memory is not treated as model tensor ownership.

---

### Requirement: Kernel Retirement Does Not Free Runtime Tensor Memory

Destroying Prepared Kernel SHALL NOT implicitly free Runtime-owned tensor
allocations.

#### Scenario: Hot swap

Given old kernel is retired

When Provider destroys native kernel state

Then model weights/KV/tensor resources remain governed by Memory Manager.