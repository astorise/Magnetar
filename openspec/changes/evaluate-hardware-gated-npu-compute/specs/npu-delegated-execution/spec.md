## ADDED Requirements

### Requirement: G04 Conditional NPU delegated execution

Intel OpenVINO/Level Zero, Qualcomm QNN and other NPU paths SHALL declare vendor/operator/dtype/shape profiles and remain hardware-gated behind Providers.

#### Scenario: G04 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a supported real NPU runs a declared model/operator profile
- **THEN** delegated compute and any explicit CPU partitions, copies and quality limits are reported

#### Scenario: G04 Ineligible or failed execution

- **WHEN** only discovery works or required operators/hardware are missing
- **THEN** compute remains Unsupported and no discovery-only result is promoted as LLM execution

### Requirement: npu-delegated-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
