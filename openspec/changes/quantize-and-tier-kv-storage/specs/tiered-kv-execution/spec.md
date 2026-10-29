## ADDED Requirements

### Requirement: D08 Qualified quantized KV storage

Quantized INT8, FP8 or Q4 KV SHALL be enabled only with compatible attention kernels, explicit scale/layout metadata and context-length quality qualification.

#### Scenario: D08 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a supported long-context generation uses quantized pages
- **THEN** memory savings and quality/latency are measured across declared context lengths

#### Scenario: D08 Ineligible or failed execution

- **WHEN** the attention implementation cannot consume the selected KV format
- **THEN** planning rejects the profile or performs an explicitly authorized conversion before use

### Requirement: D09 Explicit KV tier movement

KV offload/resume across GPU, RAM and SSD SHALL use bounded, completion-aware transfers with declared costs, integrity checks and optional measured prefetch.

#### Scenario: D09 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a session resumes from an admitted lower-tier KV copy
- **THEN** fresh Device leases are bound and transfer cost is compared with re-prefill

#### Scenario: D09 Ineligible or failed execution

- **WHEN** a page is corrupted, unavailable or still being written
- **THEN** restore fails atomically without reusing stale KV and an authorized re-prefill or error is reported

### Requirement: tiered-kv-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
