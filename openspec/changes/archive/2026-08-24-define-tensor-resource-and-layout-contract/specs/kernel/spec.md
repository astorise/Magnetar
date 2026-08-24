## ADDED Requirements

### Requirement: Kernels Receive Runtime Tensor Resource References

Kernels SHALL receive Runtime-created resource references rather than public raw pointers.

#### Scenario: Dispatch kernel

Given Kernel Invocation is created

When Provider receives it

Then it receives validated resource references and metadata.

---

### Requirement: Kernel Dispatch Validates Tensor Metadata

Kernel Dispatch SHALL validate Tensor Resource shape, dtype, layout, memory class, readiness, aliasing, mutability, and Resource Affinity before execution.

#### Scenario: Tensor not ready

Given input Tensor Resource is pending transfer

When Kernel dispatch validates inputs

Then dispatch is delayed, rejected, or replanned according to policy.

---

### Requirement: Kernel Results Update Tensor Metadata

Kernel Results SHALL update Tensor Resource readiness, residency, Resource Affinity, aliasing, and lifecycle metadata where relevant.

#### Scenario: Output tensor produced

Given Kernel writes output

When dispatch completes

Then Runtime marks output Tensor Resource ready.