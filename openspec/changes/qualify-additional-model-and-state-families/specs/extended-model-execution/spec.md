## ADDED Requirements

### Requirement: F01 Real dense family qualification

Model Components and loaders SHALL qualify a second dense family such as Llama or Gemma and additional Qwen versions using real checkpoints, tokenizer, RoPE, graph and logit conformance.

#### Scenario: F01 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a declared family profile executes a real checkpoint
- **THEN** family-specific semantics are supplied through portable graph/model Capabilities and compared with a trusted reference

#### Scenario: F01 Ineligible or failed execution

- **WHEN** configuration identifies an unsupported architecture variant
- **THEN** ingestion or Component binding rejects it without guessing compatibility from a family name

### Requirement: F04 Specialized attention and recurrent state

Specialized Components SHALL express MLA, GQA, SWA, sparse attention and KDA/recurrent semantics through validated portable operators and state schemas with explicit supported profiles.

#### Scenario: F04 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a declared specialized checkpoint generates across state boundaries
- **THEN** position/window/compression/recurrent updates and quality match its reference

#### Scenario: F04 Ineligible or failed execution

- **WHEN** a Provider lacks the operator or the state schema is incompatible
- **THEN** planning rejects the profile without embedding model-family or vendor code in Runtime Core

### Requirement: extended-model-execution promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
