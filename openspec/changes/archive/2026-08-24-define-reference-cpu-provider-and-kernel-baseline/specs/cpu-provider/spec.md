## ADDED Requirements

### Requirement: Reference CPU Provider

Magnetar SHALL define a Reference CPU Provider as a correctness-first baseline
Provider for inference execution.

#### Scenario: Register Reference CPU Provider

Given Runtime starts with Reference CPU enabled

When Provider registration runs

Then Runtime registers the Reference CPU Provider if policy allows it.

---

### Requirement: Reference CPU Provider Is Not Optimized Provider

Reference CPU Provider SHALL prioritize correctness and conformance over
performance.

#### Scenario: Slow implementation

Given a reference attention kernel is slow

When conformance validates correctness

Then it remains acceptable for the reference baseline.

---

### Requirement: Reference CPU Provider Is Not Model Architecture

Reference CPU Provider SHALL not represent model families.

#### Scenario: Qwen graph on CPU

Given a Qwen Model Component emits portable Operators

When Runtime executes on CPU

Then Reference CPU Provider executes compatible Operators and is not a
QwenProvider.

---

### Requirement: Reference CPU Provider Identity

Reference CPU Provider SHALL expose stable Provider identity.

#### Scenario: Inspect provider

Given Reference CPU Provider is registered

When Runtime lists Providers

Then stable redacted provider metadata is returned.

---

### Requirement: Reference CPU Device

Reference CPU Provider SHALL expose at least one CPU Device.

#### Scenario: CPU Device available

Given Reference CPU Provider is ready

When Runtime lists Devices

Then at least one CPU Device is visible through Runtime-owned metadata.

---

### Requirement: Host Memory Execution

Reference CPU Provider SHALL primarily operate on host memory.

#### Scenario: Device memory input

Given an input tensor is Device-only

When Reference CPU Kernel is considered

Then Runtime requires explicit movement or rejects dispatch according to policy.

---

### Requirement: Reference CPU Layout Support

Reference CPU Provider SHALL declare supported layouts explicitly.

#### Scenario: Paged layout unsupported

Given an invocation requires paged KV layout

And Reference CPU Provider does not support it

When Kernel selection runs

Then the CPU Kernel is not selected.

---

### Requirement: Reference CPU DType Support

Reference CPU Provider SHALL declare dtype support explicitly.

No silent dtype conversion SHALL occur.

#### Scenario: BF16 unsupported

Given BF16 execution is requested

And Reference CPU Provider supports only f32 compute

When dispatch is planned

Then Runtime rejects or inserts explicit dtype conversion according to policy.

---

### Requirement: Reference CPU Kernel Advertisements

Reference CPU Provider SHALL advertise only implemented Kernels.

#### Scenario: Missing kernel

Given GELU is not implemented

When Runtime queries Kernel Registry

Then no GELU CPU Kernel is assumed unless advertised.

---

### Requirement: Reference CPU Matmul Baseline

Reference CPU Provider SHALL implement correctness-first matmul for supported
dtypes and layouts.

#### Scenario: Matmul known output

Given a small matmul fixture

When CPU matmul runs

Then output matches the reference expectation within tolerance.

---

### Requirement: Reference CPU Embedding Baseline

Reference CPU Provider SHALL implement correctness-first embedding lookup.

#### Scenario: Token out of range

Given token ID exceeds vocabulary size

When embedding runs

Then execution fails with a structured error.

---

### Requirement: Reference CPU RMSNorm Baseline

Reference CPU Provider SHALL implement correctness-first RMSNorm.

#### Scenario: RMSNorm fixture

Given RMSNorm input, weight, and epsilon

When CPU RMSNorm runs

Then output matches expected values within tolerance.

---

### Requirement: Reference CPU RoPE Baseline

Reference CPU Provider SHALL implement or define a valid placeholder for RoPE.

#### Scenario: Unsupported RoPE variant

Given a RoPE variant is unsupported

When dispatch is planned

Then Runtime returns structured unsupported error.

---

### Requirement: Reference CPU Attention Baseline

Reference CPU Provider SHALL implement correctness-first attention where in
scope.

#### Scenario: Causal attention

Given causal attention fixture

When CPU attention runs

Then future tokens are masked according to causal semantics.

---

### Requirement: Reference CPU Softmax Baseline

Reference CPU Provider SHALL implement numerically stable softmax where
feasible.

#### Scenario: Softmax axis

Given softmax request has invalid axis

When validation runs

Then execution fails with structured shape or attribute error.

---

### Requirement: Reference CPU Activation Baseline

Reference CPU Provider SHALL implement supported activation kernels such as
SiLU.

#### Scenario: SiLU fixture

Given input tensor values

When CPU SiLU runs

Then output matches expected values within tolerance.

---

### Requirement: Reference CPU Elementwise Baseline

Reference CPU Provider SHALL implement supported elementwise kernels such as
add, mul, and residual-add.

#### Scenario: Shape mismatch

Given incompatible elementwise shapes

When CPU add runs

Then execution fails with structured shape error.

---

### Requirement: Reference CPU Explicit Conversion

Reference CPU Provider SHALL avoid silent dtype or layout conversion.

#### Scenario: Conversion required

Given Kernel requires f32 but input is f16 storage

When Runtime permits conversion

Then graph planning inserts explicit dtype conversion.

---

### Requirement: Reference CPU Quantization Placeholder

Unsupported quantization features SHALL fail explicitly.

#### Scenario: Unsupported quantized matmul

Given quantized matmul is requested

And no CPU quantized kernel is advertised

When selection runs

Then Runtime returns kernel candidate not found or quantization unsupported.

---

### Requirement: Reference CPU Memory Manager Integration

Reference CPU Provider SHALL use Runtime resource references and Memory Manager
accounting for Runtime-visible resources.

#### Scenario: Output tensor

Given CPU Kernel writes output

When dispatch completes

Then Memory Manager records output readiness and host residency.

---

### Requirement: Reference CPU Registry Integration

Reference CPU Kernels SHALL enter use through Kernel Registry validation.

#### Scenario: CPU kernel advertised

Given Reference CPU Provider advertises matmul

When Runtime validates advertisement

Then matmul becomes a Kernel Registry candidate.

---

### Requirement: Reference CPU Runtime Dispatch

Reference CPU Kernels SHALL execute only Runtime-created Kernel Invocations.

#### Scenario: Direct call denied

Given a Component attempts to invoke CPU Kernel directly

When Runtime enforces boundary

Then direct invocation is denied.

---

### Requirement: Reference CPU Fallback Is Explicit

Reference CPU fallback SHALL be used only when Runtime policy allows it.

#### Scenario: Fallback denied

Given optimized Provider Kernel is unavailable

And CPU fallback is disabled

When Kernel selection runs

Then Runtime rejects instead of silently using CPU.

---

### Requirement: Reference CPU Conformance Role

Reference CPU Provider SHALL support conformance fixtures for portable Operator
semantics.

#### Scenario: Generate reference output

Given a small matmul fixture

When conformance runs

Then Reference CPU Provider may generate or validate reference output.

---

### Requirement: Browser-Compatible CPU Provider Contract

Reference CPU Provider contract SHALL be platform-neutral but SHALL not require
browser implementation.

#### Scenario: Browser target

Given browser Runtime lacks Reference CPU Provider

When CPU Provider is requested

Then Runtime reports reference-cpu-browser-feature-unsupported or provider
unavailable.

---

### Requirement: Reference CPU Error Categories

Reference CPU Provider failures SHALL use structured error categories.

#### Scenario: Unsupported layout

Given a CPU Kernel receives unsupported layout

When validation runs

Then Runtime returns reference-cpu-layout-unsupported or kernel-layout-unsupported.

---

### Requirement: Reference CPU Observability

Runtime SHALL emit Reference CPU observations.

Observability SHALL not expose raw tensor values, prompts, model weights, KV
cache contents, memory pointers, Provider handles, or Device handles by default.

#### Scenario: CPU dispatch failed

Given CPU Kernel dispatch fails

When observability records it

Then Runtime emits redacted structured error metadata.