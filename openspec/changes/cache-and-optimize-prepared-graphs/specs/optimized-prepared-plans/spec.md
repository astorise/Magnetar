## ADDED Requirements

### Requirement: C02 Revalidated prepared plan cache

Prepared plans SHALL be cached with graph, shape, dtype, layout, model, Provider, Device and version fingerprints, bounded storage and validation at every reuse.

#### Scenario: C02 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** a compatible request reuses a cached plan
- **THEN** the plan is reused only after its resources, trust generation and capacity assumptions are checked

#### Scenario: C02 Ineligible or failed execution

- **WHEN** a model, driver, allocation generation or resource lifetime changes
- **THEN** the entry is invalidated and preparation or a validated fallback occurs before submission

### Requirement: C03 Validated graph optimization passes

The graph pipeline SHALL type/layout-check propagation, constant folding, dead computation removal and qualified pattern fusion while preserving outputs, side effects, KV writes and portable Component authority.

#### Scenario: C03 Qualified execution

- **GIVEN** an explicitly selected profile with its required dependencies and authorized resources
- **WHEN** an optimized graph passes semantic validation
- **THEN** it produces equivalent logits and lifecycle effects through the existing PreparedExecutionPlan path

#### Scenario: C03 Ineligible or failed execution

- **WHEN** a pass removes a required KV write or changes aliasing
- **THEN** validation rejects the candidate and publishes no partial optimized graph

### Requirement: optimized-prepared-plans promotion gate

This profile SHALL remain proposed, contract-only or experimental until all covered action gates pass on an appropriate real checkpoint and real target hardware. Runtime SHALL preserve trust, authoritative ResourceAffinity and MemoryManager admission before optimization ranking. Components SHALL NOT select native implementations or receive native handles. New paths SHALL have a disable control and a qualified fallback or explicit Unsupported outcome, and SHALL NOT grant ambient external authority.

#### Scenario: Evidence is incomplete

- **WHEN** only a contract, fixture, simulated Device, discovery result or hardware skip is available
- **THEN** the support matrix does not label this profile production-validated or claim a measured end-to-end speedup

#### Scenario: Performance conflicts with eligibility

- **WHEN** a proposed faster path violates trust, affinity, memory admission or Component authority
- **THEN** it is rejected before dispatch regardless of measured speed
