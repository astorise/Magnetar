## ADDED Requirements

### Requirement: Runtime Owns Kernel Selection

Runtime SHALL own final Kernel selection policy across Providers and Devices.

#### Scenario: Multiple Providers

Given CPU and GPU Kernels are eligible

When Runtime executes graph

Then Runtime policy chooses candidate.

---

### Requirement: Runtime Applies Eligibility Before Optimization

Runtime SHALL reject candidates violating hard constraints before ranking.

#### Scenario: Memory rejection

Given Memory Manager rejects candidate workspace

When latency ranking runs

Then candidate is absent from ranked eligible set.

---

### Requirement: Runtime Supports Selection Profiles

Runtime SHALL support policy-selected optimization profiles.

#### Scenario: Throughput workload

Given deployment requests throughput profile

When model executes

Then Runtime uses throughput-oriented ranking.

---

### Requirement: Runtime Supports Reproducible Selection

When a pin is active, Runtime SHALL NOT opportunistically substitute another Kernel; Runtime SHOULD support pinned/reproducible Kernel selection.

#### Scenario: Reproducible deployment

Given Model Instance pins Kernel artifact digest

When compatible environment executes

Then Runtime does not opportunistically switch to another kernel.

---

### Requirement: Runtime Prevents Selection Flapping

Runtime's stability policy SHALL be deterministic given identical inputs; Runtime SHOULD apply hysteresis or an equivalent stability policy.

#### Scenario: Device pressure oscillates

Given two kernels alternate by negligible score difference

When pressure changes slightly

Then Runtime avoids rapid repeated switching.

---

### Requirement: Runtime Selection Is Observable

Emitted candidate selection decision information SHALL exclude native handles and raw tensor data; Runtime SHOULD emit this redacted information.

#### Scenario: Fallback selected

Given preferred candidate is unavailable

When fallback occurs

Then Runtime records reason and selected fallback.