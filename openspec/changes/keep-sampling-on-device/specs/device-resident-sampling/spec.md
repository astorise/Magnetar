## ADDED Requirements

### Requirement: B10 Device resident logits and sampling

Provider-assisted sampling SHALL keep supported logits processing and token selection on Device, preserve ordered processors, RNG and stop semantics, and transfer only policy-required outputs.

#### Scenario: B10 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a seeded multi-step generation uses device sampling
- **THEN** token decisions and quality satisfy the reference profile and traces show no full-logit host round trip in decode

#### Scenario: B10 Ineligible or failed execution

- **WHEN** a requested processor is unsupported on Device
- **THEN** the planner explicitly uses an allowed reference path or rejects before starting the request

### Requirement: device-resident-sampling promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
