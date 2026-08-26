## ADDED Requirements

### Requirement: Post-Baseline Layouts Use Tensor Layout Contract

Blocked, paged, packed-quantized, attention-specific, provider-owned opaque, and WebGPU layouts SHALL be represented through Tensor Layout metadata.

#### Scenario: Paged KV layout

Given Provider uses paged KV cache

When Runtime tracks Tensor Resource

Then layout metadata declares paged layout without raw page pointers.

---

### Requirement: Post-Baseline Tensor Handles Remain Opaque

Post-baseline Providers SHALL not expose native tensor handles through Runtime
public APIs.

#### Scenario: Metal buffer requested

Given caller requests Metal buffer handle

When Runtime validates request

Then access is denied.