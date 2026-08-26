## ADDED Requirements

### Requirement: Post-Baseline Provider Conformance

Conformance SHALL support Provider-specific profiles for optimized and
hardware-specific Providers.

#### Scenario: CUDA conformance

Given CUDA Provider is available

When CUDA conformance profile runs

Then it validates Provider, Kernel, Tensor, Memory, Operator, and observability
contracts.

---

### Requirement: Reference Comparison

Any comparison of optimized Provider output against Reference CPU fixtures SHALL use a declared tolerance profile.
Optimized Provider conformance MAY compare outputs against Reference CPU
fixtures within declared tolerance.

#### Scenario: Optimized matmul comparison

Given optimized matmul output is produced

When compared to Reference CPU output

Then difference must be within tolerance.

---

### Requirement: Benchmark Separation

Benchmarks SHALL be reported separately from correctness conformance.

#### Scenario: Benchmark fast but wrong

Given benchmark passes performance target

But correctness conformance fails

Then Provider is not accepted as conformant.