## ADDED Requirements

### Requirement: E10 Precompiled constrained generation

JSON Schema, regex and grammar constraints SHALL compile outside token dispatch into bounded validated state and preserve allowed-token, EOS and failure semantics with qualified GPU masking when useful.

#### Scenario: E10 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a supported constraint governs generation
- **THEN** every emitted token follows the grammar and output validates against the declared schema subset

#### Scenario: E10 Ineligible or failed execution

- **WHEN** a grammar is unsupported, exceeds compile/state budgets or leaves no valid token
- **THEN** the request returns a structured error without unconstrained fallback or decode-loop compilation

### Requirement: constrained-generation promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
