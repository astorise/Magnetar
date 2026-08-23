## ADDED Requirements

### Requirement: Memory Manager Owns KV Cache Memory

Memory Manager SHALL allocate, track, admit, pressure-score, and release KV
cache memory.

#### Scenario: Allocate KV cache

Given generation requires KV cache memory

When Runtime plans generation

Then Memory Manager admits or rejects KV cache allocation.

---

### Requirement: Memory Manager Tracks KV Cache Residency

Memory Manager SHALL track KV cache residency.

Residency MAY include host memory, device memory, provider-owned memory,
browser linear memory, or future WebGPU buffers.

#### Scenario: Provider-owned residency

Given a Provider creates KV cache in native memory

When Runtime records the cache

Then Memory Manager tracks provider-owned residency metadata.

---

### Requirement: Memory Manager Handles KV Cache Pressure

Memory Manager SHALL include KV cache in memory pressure accounting.

#### Scenario: KV cache pressure high

Given KV cache memory usage is high

When Runtime evaluates memory pressure

Then KV cache pressure contributes to admission and eviction policy.

---

### Requirement: Memory Manager Releases Evicted Cache

When KV cache is evicted or released, Memory Manager SHALL release associated
resources according to ownership policy.

#### Scenario: Evict cache

Given a KV cache uses Device memory

When Runtime evicts the cache

Then Memory Manager releases or invalidates the Device memory record.
