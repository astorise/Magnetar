## ADDED Requirements

### Requirement: Reference CPU Remains Baseline After Optimized CPU

Optimized CPU Provider SHALL not replace Reference CPU as correctness baseline.

#### Scenario: Optimized CPU added

Given optimized CPU Provider is available

When conformance generates reference outputs

Then Reference CPU remains the baseline unless policy explicitly selects another
approved reference.

---

### Requirement: Optimized CPU Provider May Add Performance Features

Any performance feature added by optimized CPU Provider SHALL preserve portable Operator semantics.
Optimized CPU Provider MAY add SIMD, BLAS, thread pools, cache-aware kernels, and
fused kernels.

#### Scenario: SIMD matmul

Given optimized CPU Provider advertises SIMD matmul

When Runtime validates it

Then it must still implement portable matmul semantics.