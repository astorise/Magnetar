## ADDED Requirements

### Requirement: B05 Conditional low precision exploration

FP8, MXFP4, INT8 and trained low-bit alternatives SHALL remain profile-gated experiments until checkpoint semantics, per-layer precision and target hardware support are jointly qualified.

#### Scenario: B05 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a matching checkpoint and Device are available for an experiment
- **THEN** the report records layer precision, saturation/error behavior and measured quality/memory/performance

#### Scenario: B05 Ineligible or failed execution

- **WHEN** the checkpoint or hardware does not support the requested format
- **THEN** the experiment remains Unsupported and higher-precision execution is preserved

### Requirement: experimental-low-precision-profiles promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
