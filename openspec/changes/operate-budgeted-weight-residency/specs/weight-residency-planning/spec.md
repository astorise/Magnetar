## ADDED Requirements

### Requirement: D02 Explicit weight residency planning

WeightResidencyPlan SHALL extend existing MemoryManager pools and ResourceAffinity with bounded resident, pinned, streamed, prefetch and evict decisions, generation identities and explicit movement.

#### Scenario: D02 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a model plan assigns weights to SSD, RAM and VRAM tiers
- **THEN** each replica or transfer has an accounted lease and affinity-compatible consumer

#### Scenario: D02 Ineligible or failed execution

- **WHEN** an eviction targets an in-flight or pinned weight
- **THEN** the transition waits or is rejected without invalidating active execution

### Requirement: D03 Measured cost aware residency

Residency policy SHALL use size, observed frequency, predicted next use, I/O and conversion costs and memory pressure with bounded decisions and auditable evidence.

#### Scenario: D03 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** competing residency choices fit the admitted budget
- **THEN** the selected policy is compared against fixed-budget alternatives using transfer and stall cost

#### Scenario: D03 Ineligible or failed execution

- **WHEN** predictions are missing or oscillate
- **THEN** a conservative bounded policy and hysteresis prevent unbounded thrashing

### Requirement: D04 Authorized bounded weight streaming

Streaming SHALL read only authorized validated local artifact ranges through bounded host/device buffers and asynchronous prefetch with explicitly supported Provider movement.

#### Scenario: D04 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a model larger than Device memory executes through a supported streaming profile
- **THEN** peak memory respects admission and I/O bytes, stalls and useful prefetch are measured

#### Scenario: D04 Ineligible or failed execution

- **WHEN** a source read fails, bytes change or the Provider lacks the required operation
- **THEN** the affected generation is rejected or safely aborted without publishing unverified weights

### Requirement: D05 Completion safe staged buffers

Double/triple buffering SHALL enforce EMPTY to INFLIGHT to READY to LEASED to RECLAIM transitions and publish READY only after successful I/O, validation and Device synchronization.

#### Scenario: D05 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a prefetched buffer finishes all required operations
- **THEN** consumers acquire the correct generation and reclamation waits for CompletionToken fences

#### Scenario: D05 Ineligible or failed execution

- **WHEN** I/O fails or cancellation races with an in-flight copy
- **THEN** the buffer never becomes READY and is quarantined or reclaimed only after completion is safe

### Requirement: D11 Pressure driven memory admission

Memory control SHALL enforce separate and total quotas for weights, KV, workspace, prefetch and captures, using gradual completion-safe evictions and explicit admission/rejection.

#### Scenario: D11 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** memory pressure rises during concurrent generation
- **THEN** admission and evictions respect reservations and report the exhausted class

#### Scenario: D11 Ineligible or failed execution

- **WHEN** no compatible eviction or transfer can meet the budget
- **THEN** new work is rejected or queued by policy without implicit migration or stealing active leases

### Requirement: weight-residency-planning promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
