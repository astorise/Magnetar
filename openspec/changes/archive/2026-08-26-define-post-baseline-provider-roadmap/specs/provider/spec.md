## ADDED Requirements

### Requirement: Providers Advertise Capabilities, Not Model Families

Providers SHALL advertise capabilities and Kernels, not model-family ownership.

#### Scenario: LlamaProvider rejected

Given Provider identifies itself as `LlamaProvider`

When Runtime validates Provider architecture

Then Runtime rejects or normalizes it as invalid Provider family metadata.

---

### Requirement: Provider Registration Does Not Imply Production Readiness

Provider registration SHALL not imply production readiness or full conformance.

#### Scenario: CUDA registered

Given CUDA Provider registers

When Runtime reports status

Then it reports readiness and conformance profile status separately.

---

### Requirement: Post-Baseline Providers Use Existing Contracts

Post-baseline Providers SHALL use Provider, Device, Kernel, Tensor, Memory,
Operator, Runtime, and Conformance contracts.

#### Scenario: WebGPU Provider

Given WebGPU Provider is implemented

When it advertises Kernels

Then advertisements use Kernel Contract and Tensor Layout metadata.