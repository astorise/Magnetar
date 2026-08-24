## ADDED Requirements

### Requirement: Reference CPU Kernels Implement Operators

Reference CPU Kernels SHALL implement portable Operators according to the Kernel
Contract.

#### Scenario: CPU RMSNorm Kernel

Given Reference CPU Provider advertises RMSNorm

When Runtime validates the Kernel

Then it is tied to the portable RMSNorm Operator.

---

### Requirement: Reference CPU Kernels Prioritize Correctness

Reference CPU Kernels SHALL prioritize correctness and conformance over
performance.

#### Scenario: Slow attention

Given CPU attention is quadratic and slow

When it matches Operator semantics

Then it is acceptable for reference execution.

---

### Requirement: Reference CPU Kernels Declare Limitations

Reference CPU Kernels SHALL declare unsupported dtype, layout, shape, memory
class, batching, cancellation, and precision limitations.

#### Scenario: Unsupported paged cache

Given CPU attention does not support paged KV cache

When Kernel metadata is advertised

Then paged cache support is absent or explicitly unsupported.