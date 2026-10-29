## ADDED Requirements

### Requirement: B01 End to end native half precision

Validated profiles SHALL preserve F16/BF16 weights, activations, intermediates and KV across ingestion, graph descriptors, preparation and execution, with explicitly declared F32 accumulation or conversion points.

#### Scenario: B01 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a qualified half-precision checkpoint generates multiple tokens
- **THEN** residency and copy traces demonstrate no mandatory global F32 weight expansion

#### Scenario: B01 Ineligible or failed execution

- **WHEN** a required operator lacks the selected dtype
- **THEN** planning returns Unsupported or selects an explicitly authorized conversion/fallback before execution

### Requirement: B02 CUDA Tensor Core matrix execution

The CUDA Provider SHALL implement qualified FP16/BF16 MatMul/GEMM using supported Tensor Core paths with declared accumulation, alignment and epilogues and a verified Reference CPU fallback.

#### Scenario: B02 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** aligned and tail-shaped matrices execute on supported CUDA hardware
- **THEN** results meet dtype tolerances and end-to-end gains and memory are measured

#### Scenario: B02 Ineligible or failed execution

- **WHEN** the Device SM, alignment or precision is unsupported
- **THEN** eligibility rejects that candidate and uses a policy-approved baseline without advertising Tensor Core execution

### Requirement: native-half-precision-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
