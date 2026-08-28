## ADDED Requirements

### Requirement: Kernel Executable Memory Is Distinct From Tensor Memory

Provider-owned executable kernel memory SHALL remain distinct from Runtime
Tensor Resource memory.

#### Scenario: CUDA module loaded

Given Provider allocates executable device memory

When tensor residency is inspected

Then executable allocation is not treated as model tensor allocation.

---

### Requirement: Kernel Preparation Does Not Transfer Tensor Ownership

Kernel preparation SHALL NOT transfer Runtime Tensor Resource ownership to
Provider.

#### Scenario: Prepared kernel references buffers at invocation

Given kernel executes using Runtime tensors

When invocation completes

Then tensor lifecycle remains controlled by Memory Manager.