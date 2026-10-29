## ADDED Requirements

### Requirement: E08 Exact speculative decoding

Speculative generation SHALL implement draft/verify with target-distribution-preserving acceptance, rejection correction, isolated provisional KV and an ordinary target fallback; n-gram lookup MAY replace a costlier draft.

#### Scenario: E08 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** draft candidates are verified by the target model
- **THEN** greedy results match target-only execution and stochastic tests validate the target distribution

#### Scenario: E08 Ineligible or failed execution

- **WHEN** drafts are rejected, unavailable or too costly at current concurrency
- **THEN** unaccepted KV is discarded and target-only generation resumes without committing draft tokens

### Requirement: E09 Adaptive compatible MTP verification

Adaptive verification length SHALL use only declared compatible MTP/draft profiles, bounded confidence/concurrency decisions and the exact verification semantics.

#### Scenario: E09 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a compatible model supplies MTP candidates under changing load
- **THEN** verification length changes within budget and net accepted-token throughput and quality are measured

#### Scenario: E09 Ineligible or failed execution

- **WHEN** the head contract is absent or speculative overhead exceeds benefit
- **THEN** MTP is disabled and exact target-only generation remains available

### Requirement: F03 Explicit draft and MTP model semantics

Model Components SHALL declare supported draft/MTP heads, candidate-token semantics, required state and verification compatibility without owning token acceptance or selecting Devices.

#### Scenario: F03 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a compatible Component supplies candidates to generic speculation
- **THEN** Runtime target verification owns commits and the Component receives only inference-scoped portable resources

#### Scenario: F03 Ineligible or failed execution

- **WHEN** a model has no compatible draft/MTP head
- **THEN** the extension is Unsupported without preventing ordinary dense or MoE execution

### Requirement: speculative-generation promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
