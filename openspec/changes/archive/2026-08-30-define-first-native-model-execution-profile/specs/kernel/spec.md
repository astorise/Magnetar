## ADDED Requirements
### Requirement: Required Operators Have Reference CPU Kernels

Every mandatory first-profile Operator SHALL have a Reference CPU Kernel path.

#### Scenario: RMSNorm dispatch

Given Qwen graph contains RMSNorm

When Registry resolves it

Then eligible Reference CPU Kernel exists.

### Requirement: Reference Kernels Are Correctness Baseline

First Reference CPU Kernels SHALL prioritize deterministic understandable
correctness over optimization.

#### Scenario: Scalar MatMul

Given unoptimized implementation is mathematically correct

When first-profile tests run

Then lack of SIMD does not fail conformance.

### Requirement: Kernels Are Selected Through Registry

Reference CPU Kernel SHALL not become an architecture-specific direct call from
Qwen execution.

#### Scenario: Attention executes

Given graph node is Attention

When Runtime executes

Then Kernel selection passes through Kernel Registry/Dispatch.

### Requirement: Prepared Kernel Lifecycle Is Used

Reference CPU Kernel execution SHALL participate in PreparedKernel contract.

#### Scenario: MatMul Plan binding

Given Kernel is selected

When Plan becomes ready

Then binding references opaque PreparedKernelId or equivalent prepared state.