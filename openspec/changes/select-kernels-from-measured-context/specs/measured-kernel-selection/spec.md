## ADDED Requirements

### Requirement: C01 Measured benchmark to selection pipeline

Runtime selection SHALL consume qualified real KernelBenchmark evidence through KernelPerformanceModel and KernelSelectionPolicy only after compatibility, trust, affinity and memory eligibility.

#### Scenario: C01 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** two eligible kernel candidates have versioned measurements
- **THEN** selection records the evidence and selects under the declared objective

#### Scenario: C01 Ineligible or failed execution

- **WHEN** a faster candidate has invalid trust, incompatible affinity or an infeasible workspace
- **THEN** it is excluded before ranking regardless of benchmark score

### Requirement: C05 Context specialized kernel selection

Selection SHALL specialize eligible candidates by prefill/decode, batch size, context length, memory pressure and hardware architecture using versioned evidence.

#### Scenario: C05 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** the workload changes from prefill to a qualified decode class
- **THEN** selection uses the appropriate evidence without changing graph or model semantics

#### Scenario: C05 Ineligible or failed execution

- **WHEN** the context is outside measured support or a preferred candidate exceeds budget
- **THEN** selection uses a qualified conservative candidate and marks extrapolated evidence explicitly

### Requirement: C09 Pinned execution and rollback controls

Operators SHALL be able to pin kernel/plan generations, disable individual optimizations and roll back at a safe boundary with diagnostic reasons and reproducible evidence.

#### Scenario: C09 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** pinned mode runs an identical qualified profile
- **THEN** adaptive retuning and noncontractual nondeterminism are disabled and selected identities are reported

#### Scenario: C09 Ineligible or failed execution

- **WHEN** a selected generation is revoked or fails at runtime
- **THEN** new work stops using it and rollback revalidates trust and resources rather than blindly reviving invalid code

### Requirement: measured-kernel-selection promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
