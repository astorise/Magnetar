## ADDED Requirements

### Requirement: F07 Conditional multimodal and embedding Components

Vision-first multimodal, embedding and reranking experiments SHALL declare separate preprocessing/tokenization and compute Capabilities, artifact authority and supported input/output schemas.

#### Scenario: F07 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a supported real model consumes authorized media or text
- **THEN** preprocessing and compute outputs match a reference without native handles entering WIT

#### Scenario: F07 Ineligible or failed execution

- **WHEN** media access is unauthorized or its shape/operator is unsupported
- **THEN** the boundary rejects it without ambient filesystem/network authority or silent text-only substitution

### Requirement: multimodal-model-experiments promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
