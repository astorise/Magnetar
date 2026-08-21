# resource-affinity Specification

## Purpose
TBD - created by archiving change define-resource-affinity-model. Update Purpose after archive.
## Requirements
### Requirement: Runtime-Native Resource Affinity

The Runtime SHALL represent Resource Affinity as immutable host-side metadata
composed only of stable identifiers, exact capability versions, artifact
fingerprints, and fallback classification.

A Provider-owned resource SHALL record its Provider binding, a device-resident
resource SHALL record its Device binding, and a resource created by a resolved
Capability SHALL record that exact Capability identifier and version.

#### Scenario: Record an opaque resource

- **GIVEN** the Runtime has selected a Provider and exact Capability version
- **WHEN** a host adapter wraps a newly created opaque resource
- **THEN** the resource affinity records the selected Provider, Capability, and
  Runtime execution context
- **AND** it records the selected Device when the resource is device-resident

### Requirement: Affinity Bindings

The Resource Affinity model SHALL provide distinct Provider, Device,
Capability, artifact, execution-context, and affinity-group binding types.

Capability bindings for live resources SHALL use exact versions. Semantic
version negotiation SHALL occur only before a resource is created.

#### Scenario: Preserve an exact live capability binding

- **GIVEN** a resource created by `magnetar:compute/run@1.1.0`
- **WHEN** the Runtime aggregates constraints for a dependent call
- **THEN** the resource remains bound to `magnetar:compute/run@1.1.0`
- **AND** the Runtime does not reinterpret it as a resource from another
  compatible Capability version

### Requirement: Artifact Binding by Role

Each artifact binding SHALL contain a role and a canonical content
fingerprint. Affinities SHALL conflict only when they declare different
fingerprints for the same role.

Related resources SHALL use an explicit shared role such as `model-bundle` or
`compatibility-manifest` when compatibility depends on common bundle identity;
the Runtime SHALL NOT compare a model digest directly with a tokenizer digest.

#### Scenario: Validate a model bundle

- **GIVEN** a loaded-model affinity and tokenizer affinity with the same
  `model-bundle` fingerprint
- **AND** distinct `model` and `tokenizer` content fingerprints
- **WHEN** the Runtime aggregates their constraints
- **THEN** the artifact bindings are compatible

#### Scenario: Reject a conflicting bundle

- **GIVEN** two resource affinities with different `model-bundle` fingerprints
- **WHEN** the Runtime aggregates their constraints
- **THEN** aggregation fails with an artifact-mismatch error identifying the
  role and both fingerprints

### Requirement: Affinity Constraint Aggregation

The Runtime SHALL aggregate all resource affinities for a dependent call before
resolving a Provider.

Aggregation SHALL reject conflicting Provider, Device, execution-context, and
affinity-group bindings. It SHALL reject different exact versions bound to the
same Capability identifier and different fingerprints bound to the same
artifact role. Distinct Capability identifiers and artifact roles SHALL be
preserved in the aggregate.

#### Scenario: Aggregate a coherent dependency chain

- **GIVEN** tensor and graph resources bound to the same Provider, Device, and
  execution context
- **WHEN** their affinities are aggregated for Compute submission
- **THEN** one coherent constraint set preserves every binding

#### Scenario: Reject resources from different Devices

- **GIVEN** two resources bound to different Device identifiers
- **WHEN** their affinities are aggregated
- **THEN** aggregation fails with a structured device-mismatch error before a
  Provider is returned to the caller

### Requirement: Affinity-Aware Capability Resolution

The Runtime SHALL provide additive affinity-aware resolution alongside the
existing stateless resolver.

Affinity-aware resolution SHALL select a single Provider and exact compatible
Capability version that satisfy the complete aggregated constraint set. When a
Provider or Device is already bound, resolution SHALL search compatible
Capability versions implemented by that Provider rather than selecting a
global version first.

The returned resolution SHALL preserve the aggregate and record the selected
Provider, exact Capability version, and Runtime execution context for the
resource to be created. A dependent resolution SHALL also preserve an existing
affinity group or create one when its dependencies are not already grouped.

#### Scenario: Resolve for Provider-bound resources

- **GIVEN** a live resource bound to a Provider that implements a compatible
  Capability version
- **WHEN** a dependent Capability is resolved
- **THEN** the Runtime returns that Provider and its best compatible version
- **AND** no other Provider is considered a fallback for the live call

#### Scenario: Bound Provider is unavailable

- **GIVEN** a live resource bound to an unavailable Provider
- **WHEN** a dependent Capability is resolved
- **THEN** resolution returns a structured bound-provider-unavailable error
- **AND** it does not silently select another Provider

### Requirement: Structured Affinity Errors

Affinity validation and resolution SHALL return stable structured error
categories for Provider, Device, Capability, artifact, execution-context, and
affinity-group mismatches, unavailable bound resources, and missing compatible
implementations.

Each mismatch SHALL retain the expected and conflicting stable values for
runtime diagnostics. These errors SHALL remain separate from Provider
lifecycle and registration failures.

#### Scenario: Report an incompatibility

- **GIVEN** a dependent call containing incompatible resource affinities
- **WHEN** validation fails
- **THEN** the caller receives the specific affinity error category and both
  conflicting stable values
- **AND** no native handle or Provider-specific object is included

### Requirement: Affinity-Bearing Resource Envelope

The Runtime SHALL provide a reusable host-side envelope that associates an
opaque Provider-owned value with its Resource Affinity without exposing the
value through the affinity descriptor.

The envelope SHALL support current Compute tensor, graph, and operation host
adapters and future opaque resource adapters without defining placeholder
model, tokenizer, template, or generation WIT contracts in this change.

#### Scenario: Carry Compute affinity

- **GIVEN** host-side tensor and graph handles returned by a Compute adapter
- **WHEN** each handle is placed in the affinity-bearing envelope
- **THEN** the Runtime can validate their affinities before submitting work
- **AND** the adapter records its authoritative Device selection when either
  handle is device-resident
- **AND** the opaque handles remain accessible only to native host code

### Requirement: Fallback Classification and Live-State Safety

Every Resource Affinity SHALL classify recovery as transparent, restartable,
or Provider-pinned.

Combining resources SHALL preserve the most restrictive classification.
Restartable SHALL require explicit recreation from replayable input, and
Provider-pinned SHALL require explicit teardown or failure. No classification
SHALL authorize the Runtime to ignore bindings on a live resource or migrate it
implicitly.

#### Scenario: Provider fails after resource creation

- **GIVEN** a live Provider-bound resource classified as restartable or
  Provider-pinned
- **WHEN** its Provider becomes unavailable
- **THEN** the current operation fails without implicit migration
- **AND** a restartable operation can proceed elsewhere only after the caller
  explicitly recreates its resources from replayable input

### Requirement: Host-Only Affinity Metadata

Resource Affinity SHALL remain Runtime and host-adapter metadata in this
change. It SHALL NOT add Provider names, Device identifiers, native pointers,
queues, streams, locks, Rust trait objects, or backend storage to a Component
WIT contract.

#### Scenario: Inspect affinity for diagnostics

- **GIVEN** a Runtime diagnostic or scheduling policy inspects Resource
  Affinity
- **WHEN** it reads the descriptor
- **THEN** it observes stable identifiers, versions, classifications, and
  fingerprints only
- **AND** the `magnetar:compute@1.1.0` WIT contract remains unchanged

