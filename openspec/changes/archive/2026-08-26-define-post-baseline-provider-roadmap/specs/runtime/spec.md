## ADDED Requirements

### Requirement: Runtime Owns Optimized Provider Selection

Runtime SHALL own optimized Provider selection through Kernel Registry,
Resource Affinity, Memory Manager, Provider readiness, and policy.

#### Scenario: User prefers CUDA

Given user preference requests CUDA

When Runtime selects execution

Then CUDA is selected only if compatible and policy allows.

---

### Requirement: Runtime Rejects Native Handle Exposure

Runtime SHALL reject attempts to expose native Provider, Device, Kernel, tensor,
or memory handles through public APIs.

#### Scenario: Device pointer diagnostic

Given diagnostic path attempts to include Device pointer

When Runtime redaction validates output

Then the pointer is removed or rejected.

---

### Requirement: Runtime Keeps Baseline Compatibility

Post-baseline Provider additions SHALL not break Reference CPU baseline
conformance.

#### Scenario: CUDA added

Given CUDA Provider is added

When CPU-only baseline conformance runs

Then Reference CPU baseline still passes.