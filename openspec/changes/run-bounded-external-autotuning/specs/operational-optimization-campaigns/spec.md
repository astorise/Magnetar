## ADDED Requirements

### Requirement: C06 Bounded off path autotuning

Autotuning SHALL run outside token dispatch on already-qualified candidates with explicit compute/memory/time budgets, canaries, hysteresis and rollback at safe generation boundaries.

#### Scenario: C06 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a tuning campaign yields a qualified improvement
- **THEN** a new generation is promoted after canary checks while in-flight work retains its original leases

#### Scenario: C06 Ineligible or failed execution

- **WHEN** canary tail latency regresses or campaign budgets are exhausted
- **THEN** promotion stops and the last valid generation remains available without blocking decode

### Requirement: C08 External optimization campaign boundary

The external Optimization Plane SHALL produce benchmark evidence, signed artifacts and recommendations; Runtime SHALL verify admission and control promotion without granting the plane inference resource or decode-loop compilation authority.

#### Scenario: C08 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a signed campaign result is proposed
- **THEN** Runtime verifies provenance, qualification and current compatibility before accepting a new generation

#### Scenario: C08 Ineligible or failed execution

- **WHEN** a recommendation is unsigned, stale, revoked or requests execution privileges
- **THEN** Runtime rejects it and continues the qualified pinned generation

### Requirement: operational-optimization-campaigns promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
