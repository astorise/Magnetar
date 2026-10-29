## ADDED Requirements

### Requirement: A06 Redacted phase attribution

Observability SHALL attribute kernel, plan, copy, I/O and cache costs through correlated phase records, use actual device timing when available and redact prompts, weight bytes, tensor values, secrets and sensitive paths.

#### Scenario: A06 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a request crosses preparation, I/O, copies and compute
- **THEN** the report distinguishes device time, host time, overlap and unavailable counters without double counting

#### Scenario: A06 Ineligible or failed execution

- **WHEN** a prompt or tensor contains a recognizable sentinel
- **THEN** logs, traces and exported metrics omit that sentinel and retain only permitted metadata

### Requirement: D01 Anonymized access traces and cache simulation

Offline tools SHALL evaluate LRU, LFU, SLRU, weighted-cost and offline-optimum cache policies from redacted tensor/expert access traces and budget ladders.

#### Scenario: D01 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a versioned trace is replayed
- **THEN** the simulator reports hit, prefetch, bytes moved, stall costs and an offline-oracle bound under identical budgets

#### Scenario: D01 Ineligible or failed execution

- **WHEN** a trace lacks complete future access information or contains sensitive identities
- **THEN** the oracle is marked unavailable and sensitive content is removed before export

### Requirement: inference-cost-attribution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
