## ADDED Requirements

### Requirement: F02 Complete MoE model execution

MoE Components SHALL represent router, top-k, expert/shared-expert graphs and lifecycles through portable Capabilities while Runtime owns residency and Providers own execution.

#### Scenario: F02 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a real MoE checkpoint performs prefill and multi-step decode
- **THEN** routing, combination weights, expert utilization and logits match the qualified reference

#### Scenario: F02 Ineligible or failed execution

- **WHEN** routing produces invalid expert identities or insufficient admitted capacity
- **THEN** validation fails or uses an explicitly supported fallback before any invalid expert access

### Requirement: F05 Trace driven expert residency

MoE expert caching SHALL use validated traces, hot-layer pinning and top-k prefetch under explicit admission and completion-safe staged buffers.

#### Scenario: F05 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a real expert access distribution drives a fixed-budget policy
- **THEN** useful hit rate, misprefetch, I/O stalls and imbalance are compared against baseline policies

#### Scenario: F05 Ineligible or failed execution

- **WHEN** routing shifts abruptly or prefetched experts are unused
- **THEN** bounded eviction/hysteresis protects active experts and prevents runaway transfer or memory use

### Requirement: B09 Quantized expert kernels

Providers SHALL qualify grouped quantized expert GEMM and fused dispatch/gather/scatter against the portable MoE routing and accumulation semantics.

#### Scenario: B09 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a supported top-k expert batch executes
- **THEN** outputs match unfused expert references including ordering, shared experts and quantization metadata

#### Scenario: B09 Ineligible or failed execution

- **WHEN** expert routing is empty, skewed or contains unsupported packed layouts
- **THEN** the supported fallback or structured rejection preserves token ownership and never reads another expert buffer

### Requirement: moe-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
