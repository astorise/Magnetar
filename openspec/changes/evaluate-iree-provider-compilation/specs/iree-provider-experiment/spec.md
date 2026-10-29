## ADDED Requirements

### Requirement: C07 Provider isolated IREE experiment

IREE/MLIR graph compilation SHALL be an opt-in Provider experiment with a bounded graph/operator subset, reproducible artifacts and a measured go/no-go against the existing implementation.

#### Scenario: C07 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a supported graph is compiled in the experiment
- **THEN** numerical parity, compile cost, steady-state gains, memory and licensing are reported before integration

#### Scenario: C07 Ineligible or failed execution

- **WHEN** the compiler cannot represent an operator or yields no justified gain
- **THEN** the experiment stops or stays experimental and portable graph contracts do not gain compiler handles

### Requirement: iree-provider-experiment promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
