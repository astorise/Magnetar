## ADDED Requirements

### Requirement: D10 Versioned inference state snapshot

Local serialization SHALL capture a consistent committed token boundary and model/component/tokenizer/adapter identities, KV or recurrent state, RNG, positions, dtype/layout, version and checksums without native handles.

#### Scenario: D10 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a compatible snapshot is restored into newly admitted resources
- **THEN** subsequent generation follows its declared numerical and RNG continuity profile

#### Scenario: D10 Ineligible or failed execution

- **WHEN** the snapshot is truncated, incompatible, untrusted or exceeds destination quotas
- **THEN** restore rejects before publishing session state and releases only safely completed allocations

### Requirement: G07 Authorized portable state export import

InferenceStateExport/Import SHALL define a versioned bounded envelope with model digests, dtype/layout, positions, component/adapter/tokenizer and state identities, integrity and explicit source/destination authorization; native handles SHALL NOT cross nodes.

#### Scenario: G07 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** an authorized destination imports a compatible snapshot
- **THEN** it validates trust and capacity and binds fresh local resources before atomically exposing the resumed session

#### Scenario: G07 Ineligible or failed execution

- **WHEN** the envelope is replayed outside its authorization, corrupted or incompatible
- **THEN** import rejects it without state publication and reports a redacted compatibility reason

### Requirement: inference-state-transfer promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
