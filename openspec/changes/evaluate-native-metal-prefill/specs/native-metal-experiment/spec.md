## ADDED Requirements

### Requirement: G03 Conditional native Metal specialization

Native Metal SHALL remain a bounded Apple Silicon experiment until a macOS machine proves specialized prefill kernels improve a qualified WGPU baseline without weakening lifecycle or numerical correctness.

#### Scenario: G03 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** real Apple hardware runs the POC and reference profiles
- **THEN** the go/no-go includes MPS or simdgroup paths actually used, gains, memory and maintenance costs

#### Scenario: G03 Ineligible or failed execution

- **WHEN** macOS hardware is absent or the POC provides no justified gain
- **THEN** the Provider remains unavailable or experimental and no native compute support is claimed

### Requirement: native-metal-experiment promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
