# compute Specification

## Purpose
TBD - created by archiving change define-runtime-memory-manager. Update Purpose after archive.
## Requirements
### Requirement: Compute Describes Tensors, Memory Manager Realizes Memory

Compute SHALL define portable tensor and operation descriptors.

The Memory Manager SHALL realize memory allocation, residency, staging, and
placement feasibility.

#### Scenario: TensorDescriptor validated

Given a TensorDescriptor declares dtype, shape, and layout

When Runtime determines where data resides

Then Memory Manager owns residency and allocation state.

---

### Requirement: Compute Does Not Own Allocator State

Compute modules SHALL NOT own caching allocator, arena, pinned memory, pending
queue, or memory pressure state.

#### Scenario: Search allocator behavior

Given a developer searches for caching allocator policy

When they inspect Compute modules

Then Compute does not own that policy.

---

### Requirement: Compute Storage And Compute DType Are Memory-Relevant

Compute descriptors SHALL preserve dtype semantics needed by memory planning.

Memory Manager SHALL own the memory implications of storage dtype and compute
dtype.

#### Scenario: INT8 storage BF16 compute

Given Compute execution uses BF16 compute from INT8 stored weights

When memory is planned

Then Memory Manager accounts for compressed storage and temporary compute
workspace.

---

### Requirement: Compute Host Staging Policy Is Enforced By Memory Manager

Compute SHALL express HostStagingPolicy when data movement policy constrains host staging.

Memory Manager SHALL enforce it during memory and data movement planning.

#### Scenario: Host staging forbidden

Given Compute data movement forbids host staging

When Memory Manager evaluates movement

Then staging through host memory is rejected.

