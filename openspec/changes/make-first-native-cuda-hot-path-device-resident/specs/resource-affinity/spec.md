## MODIFIED Requirements

### Requirement: Runtime-Native Resource Affinity

The Runtime SHALL represent Resource Affinity as immutable host-side metadata
composed only of stable identifiers, exact capability versions, artifact
fingerprints, and fallback classification.

A Provider-owned resource SHALL record its Provider binding, a device-resident
resource SHALL record its Device binding, and a resource created by a resolved
Capability SHALL record that exact Capability identifier and version.

Runtime SHALL enforce, in production builds (not only development-time
assertions), that a Kernel invocation's resolved Provider and Device agree
with the `ResourceAffinity` recorded for every resource that invocation
reads or writes; a divergence SHALL be rejected with a structured error
before the invocation is submitted to a Provider.

#### Scenario: Record an opaque resource

- **GIVEN** the Runtime has selected a Provider and exact Capability version
- **WHEN** a host adapter wraps a newly created opaque resource
- **THEN** the resource affinity records the selected Provider, Capability, and
  Runtime execution context
- **AND** it records the selected Device when the resource is device-resident

#### Scenario: Reject a Provider/Device mismatch at dispatch

- **GIVEN** a Kernel invocation resolved to one Provider and Device
- **AND** a resource attached to that invocation records a different
  Provider or Device in its `ResourceAffinity`
- **WHEN** the invocation is validated before submission
- **THEN** the Runtime rejects it with a structured Provider/Device-mismatch
  error, in every build configuration, not only when debug assertions are
  enabled

### Requirement: Affinity Constraint Aggregation

The Runtime SHALL aggregate all resource affinities for a dependent call before
resolving a Provider.

Aggregation SHALL reject conflicting Provider, Device, execution-context, and
affinity-group bindings. It SHALL reject different exact versions bound to the
same Capability identifier and different fingerprints bound to the same
artifact role. Distinct Capability identifiers and artifact roles SHALL be
preserved in the aggregate.

A resource that already carries a recorded `ResourceAffinity` (for example,
one bound at Model Load time) SHALL keep that affinity when it participates
in a later dependent call; the Runtime SHALL aggregate the dispatch's own
affinity with the resource's existing one through this same aggregation
contract, and SHALL NOT silently replace the resource's recorded affinity
with a freshly-derived one.

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

#### Scenario: A Resident resource's own affinity survives a later dispatch

- **GIVEN** a resource (for example, a Model Load-time weight) already
  carries a recorded `ResourceAffinity` for Provider X and Device Y
- **WHEN** a later dispatch on the same Provider and Device consumes that
  resource
- **THEN** the resource's recorded affinity is aggregated with, not replaced
  by, the dispatch's own freshly-derived affinity
- **AND** the resource's original Provider/Device binding remains present
  in the aggregate afterward
