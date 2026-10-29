## ADDED Requirements

### Requirement: E03 Asynchronous CPU GPU scheduling

Runtime scheduling SHALL overlap preparation of batch N+1 with forward execution of N using immutable metadata generations, explicit dependencies and completion barriers.

#### Scenario: E03 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** independent schedule preparation overlaps device work
- **THEN** trace evidence demonstrates overlap with correct output attribution and bounded buffers

#### Scenario: E03 Ineligible or failed execution

- **WHEN** cancellation, metadata mutation or a failed completion races with overlap
- **THEN** dependent submissions are withheld and resources remain protected until failure resolution

### Requirement: E04 Batch class graph replay

Serving SHALL bind CUDA replay to admitted batch classes with bounded padding, valid KV/weight leases, observable cache behavior and an uncaptured fallback.

#### Scenario: E04 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a request batch fits a qualified replay class
- **THEN** padding, useful token work, capture memory and replay benefit are measured

#### Scenario: E04 Ineligible or failed execution

- **WHEN** the batch changes shape, loses a lease or exceeds a class budget
- **THEN** the class is invalidated or uncaptured execution is selected without stale binding reuse

### Requirement: asynchronous-batch-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
