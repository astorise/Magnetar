## ADDED Requirements

### Requirement: Post-Baseline Memory Classes Are Tracked

Device, pinned-host, unified, shared, provider-owned, browser-linear-memory, and WebGPU buffer memory classes SHALL be tracked by Memory Manager when supported.

#### Scenario: CUDA device output

Given CUDA Kernel writes Device output

When dispatch completes

Then Memory Manager tracks memory class, residency, and Resource Affinity.

---

### Requirement: Provider Data Movement Is Explicit

Post-baseline Provider data movement SHALL be explicit and policy-controlled.

#### Scenario: Device to host fallback

Given fallback to CPU requires Device-to-host transfer

When planning runs

Then Runtime inserts explicit movement or rejects fallback.