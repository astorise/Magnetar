## ADDED Requirements

### Requirement: Post-Baseline Provider Roadmap

Magnetar SHALL define a Provider roadmap for optimized and hardware-specific
Providers after the Reference CPU baseline.

#### Scenario: Roadmap exists

Given baseline implementation is complete

When post-baseline planning begins

Then Provider phases and gates are defined.

---

### Requirement: Reference CPU Remains Correctness Baseline

Reference CPU Provider SHALL remain the correctness baseline for small
deterministic fixtures.

#### Scenario: Optimized output comparison

Given optimized Provider output differs from Reference CPU beyond tolerance

When conformance runs

Then optimized Provider conformance fails.

---

### Requirement: No Model-Family Providers

Post-baseline roadmap SHALL NOT introduce model-family Providers.

#### Scenario: QwenProvider attempted

Given implementation introduces `QwenProvider`

When roadmap validation runs

Then validation rejects it.

---

### Requirement: Provider Roadmap Phases

Each defined Provider roadmap phase SHALL declare required conformance gates before that phase is considered production-ready.
Provider roadmap SHOULD define phases for optimized CPU, CUDA, Metal, OpenVINO,
QNN, WebGPU, quantized execution, advanced attention, and performance profiles.

#### Scenario: CUDA planned

Given CUDA support is requested

When roadmap is inspected

Then CUDA appears as a Provider phase with conformance gates.

---

### Requirement: Optimized Providers Preserve Portable Semantics

Optimized Providers SHALL not redefine portable Operator semantics.

#### Scenario: Fused RMSNorm

Given optimized Provider implements fused RMSNorm

When conformance runs

Then output matches RMSNorm semantics within tolerance.

---

### Requirement: Native Handles Remain Hidden

Post-baseline Providers SHALL not expose native Provider, Device, Kernel, memory,
or framework handles through public APIs.

#### Scenario: CUDA handle requested

Given caller asks for CUDA device pointer

When Runtime validates request

Then access is denied.

---

### Requirement: Advanced Attention Is Explicit

Advanced attention support SHALL be explicit and conformance-gated.

#### Scenario: Flash attention unsupported

Given Provider lacks flash attention

When graph requires flash attention

Then Runtime fails explicitly or uses validated fallback according to policy.

---

### Requirement: Quantized Execution Is Explicit

Quantized execution SHALL require explicit metadata and conformance.

#### Scenario: Hidden dequantization

Given quantized tensor is silently dequantized without graph plan

When conformance validates execution

Then conformance fails.

---

### Requirement: Layout Expansion Is Explicit

Specialized layouts SHALL be represented through Tensor Layout metadata.

#### Scenario: Packed layout

Given Provider requires packed quantized layout

When graph planning runs

Then layout requirement is explicit.

---

### Requirement: Memory Expansion Is Tracked

Post-baseline memory classes SHALL be tracked by Memory Manager.

#### Scenario: Device output

Given CUDA Provider creates Device-resident output

When dispatch completes

Then Memory Manager tracks residency and Resource Affinity.

---

### Requirement: Provider Conformance Profiles

Each post-baseline Provider SHALL declare conformance profiles.

#### Scenario: Provider registered

Given CUDA Provider registers

When Runtime inspects it

Then Provider readiness does not imply all conformance profiles passed.

---

### Requirement: Benchmarks Do Not Replace Conformance

Performance benchmarks SHALL be separate from correctness conformance.

#### Scenario: Fast incorrect kernel

Given optimized kernel is fast but incorrect

When conformance runs

Then Provider fails regardless of benchmark performance.

---

### Requirement: Fallback Remains Explicit

Fallback across Providers SHALL remain explicit and policy-controlled.

#### Scenario: CUDA fails

Given CUDA Kernel fails

When CPU fallback is disabled

Then Runtime fails instead of silently falling back.

---

### Requirement: Runtime API Stability

Post-baseline Providers SHALL not require basic Runtime Inference API to expose
Provider-specific handles.

#### Scenario: Metal Provider enabled

Given Metal Provider is available

When inference API is used

Then caller still uses Provider-independent inference request.

---

### Requirement: CLI Boundary Stability

Provider roadmap SHALL preserve Runtime-owned Provider selection.

#### Scenario: CLI preference

Given CLI requests Provider preference

When Runtime executes inference

Then Runtime treats it as non-authoritative policy input.

---

### Requirement: Provider Roadmap Error Categories

Provider roadmap failures SHALL use structured error categories.

#### Scenario: Provider feature unsupported

Given requested advanced attention is unsupported

When planning runs

Then Runtime returns provider-feature-unsupported or specific structured error.

---

### Requirement: Provider Roadmap Observability

Provider roadmap observations SHALL NOT expose native handles or raw tensor values.
Runtime SHOULD emit Provider roadmap observations with default redaction.

#### Scenario: Fallback denied

Given fallback is denied by policy

When observability records it

Then no native handles or raw tensor values are logged.