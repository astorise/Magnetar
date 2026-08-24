## ADDED Requirements

### Requirement: Runtime Owns Tensor Resource Identity

Runtime SHALL own Tensor Resource IDs, lookup, authorization, lifecycle, and metadata safety.

#### Scenario: Forged tensor ID

Given caller provides fabricated TensorResourceId

When Runtime resolves it

Then Runtime rejects it as not found or unauthorized.

---

### Requirement: Runtime Prevents Raw Tensor Exposure

Runtime SHALL not expose raw tensor pointers, native handles, allocation addresses, Provider internals, Device internals, raw KV cache contents, raw model weights, or raw prompts through Tensor APIs by default.

#### Scenario: Tensor metadata

Given a caller requests Tensor metadata

When Runtime responds

Then it returns stable metadata only.

---

### Requirement: Runtime Requires Explicit Tensor Conversion

Runtime SHALL require explicit planning for dtype conversion, layout conversion, memory movement, host staging, quantization, dequantization, and opaque materialization.

#### Scenario: CPU fallback

Given fallback to CPU requires Device-to-host movement

When planning runs

Then Runtime inserts explicit movement or rejects fallback according to policy.

---

### Requirement: Runtime Observes Tensor Lifecycle

Runtime SHOULD emit structured Tensor observations, and observations SHALL not expose raw data or handles.

#### Scenario: Tensor released

Given Tensor Resource is released

When Runtime emits observability

Then it records redacted resource metadata.