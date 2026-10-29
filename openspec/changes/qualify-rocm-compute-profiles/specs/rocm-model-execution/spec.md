## ADDED Requirements

### Requirement: G02 ROCm real compute qualification

ROCm/HIP SHALL advertise only hardware-qualified compute profiles through generic Provider execution, with numerical, memory and lifecycle conformance on real AMD hardware.

#### Scenario: G02 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a supported AMD Device executes the reference checkpoint
- **THEN** kernel dispatch and generation evidence demonstrate real HIP compute with declared dtypes

#### Scenario: G02 Ineligible or failed execution

- **WHEN** HIP discovery succeeds but no matching compute profile is qualified
- **THEN** discovery is reported separately and unsupported compute is not advertised

### Requirement: rocm-model-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
