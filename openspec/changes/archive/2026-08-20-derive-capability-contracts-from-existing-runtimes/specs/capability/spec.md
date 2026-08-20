## ADDED Requirements

### Requirement: Evidence-Based Capability Derivation

Every Capability family proposed by a source-derived taxonomy SHALL identify
the pinned external runtime revision, relevant source path, and source symbol
that motivate its boundary.

#### Scenario: Trace a proposed family

- **WHEN** a reviewer inspects a Capability family proposed by the Candle and
  Crane source-derived taxonomy
- **THEN** the reviewer can trace it to at least one Candle or Crane source
  symbol at a recorded revision

### Requirement: Layered Responsibility Taxonomy

The contract-derivation taxonomy SHALL classify source responsibilities into
low-level execution, model-level execution, or application-level abilities and
SHALL map each responsibility to Magnetar's Provider, Capability, Component,
Device, or runtime-service roles.

#### Scenario: Classify a runtime responsibility

- **WHEN** an existing runtime responsibility is mapped into Magnetar
- **THEN** it has one primary taxonomy layer, a target Magnetar role, stated
  responsibilities, and explicit exclusions

### Requirement: Capability Candidate Qualification

A responsibility SHALL be proposed as a Capability family only when it has a
portable WIT boundary; native-only responsibilities SHALL remain Provider,
Device, or runtime implementation details.

#### Scenario: Classify a native backend surface

- **WHEN** a source responsibility depends on native resource representation
  and has no portable contract boundary
- **THEN** it is excluded from the Capability candidates and assigned to the
  appropriate native Magnetar role

### Requirement: Capability Dependency Documentation

Every proposed Capability family SHALL document the other Capability families
or existing Magnetar runtime services required for its execution.

#### Scenario: Review a candidate contract

- **WHEN** a candidate contract is prepared for a future WIT change
- **THEN** its transitive execution dependencies are visible from the taxonomy

### Requirement: Native and Component Boundary Classification

Every analyzed responsibility SHALL be classified as native-only,
Component-suitable, or hybrid. Hybrid candidates SHALL distinguish their WIT
surface from Provider-owned resources, with rationale covering ownership, data
transfer, state, and call granularity.

#### Scenario: Select a contract boundary

- **WHEN** a responsibility owns device resources or crosses a portable
  Component boundary
- **THEN** the taxonomy explains which part remains native and which part can be
  expressed through WIT

### Requirement: Fallback Semantics Classification

Every proposed Capability family SHALL document whether fallback is
transparent, restartable from replayable input, or pinned for the lifetime of
Provider-owned state.

#### Scenario: Provider failure during execution

- **WHEN** a Provider fails before or during a Capability operation
- **THEN** the taxonomy states whether another compatible Provider can be used
  without violating resource ownership or observable stream semantics

### Requirement: Provisional WIT Package Mapping

Component-suitable and hybrid Capability families SHALL be mapped to candidate
WIT packages without registering those candidates as stable runtime contracts.

#### Scenario: Prepare a follow-up WIT proposal

- **WHEN** a future change selects a Capability family for standardization
- **THEN** it has a provisional package boundary and the unresolved contract
  decisions needed before versioning

### Requirement: Magnetar-Native Contract Shape

Derived contracts SHALL express Magnetar's Provider, Capability, Component,
and Device model rather than copying Candle or Crane APIs.

#### Scenario: Adapt an existing interface

- **WHEN** a Candle or Crane interface motivates a Magnetar boundary
- **THEN** Rust-specific generics, callbacks, storage implementations, and
  concrete model types are excluded from the portable contract shape
