# Tighten Compute Component Boundary

## Why

Magnetar's canonical architecture requires portable Components to request
Capabilities while the Runtime resolves concrete Providers and Devices.

The canonical execution relationship is:

```text
Component
    |
    | imports Capability
    v
Runtime
    |
    | Resolution Policy
    | Resource Affinity
    v
Provider
    |
    v
Device
```

The current `magnetar:compute/run@1.1.0` contract mostly follows this model.

It already provides:

- coarse graph submission
- portable tensor descriptors
- opaque tensor resources
- opaque graph resources
- structured compute errors
- explicit data movement semantics
- Provider-independent operation schemas

However, the current portable data-movement descriptor exposes concrete
execution-routing fields:

```wit
record data-movement-descriptor {
    kind: data-movement-kind,
    source: data-movement-source,
    output: tensor-descriptor,
    target-provider: option<string>,
    target-device: option<string>,
    target-affinity-group: option<u64>,
    allow-host-staging: bool,
}
```

These fields allow a portable Component to express:

```text
use Provider X
use Device Y
use Runtime affinity group Z
```

That contradicts the established architecture.

Provider selection belongs to Resolution Policy.

Device selection belongs to Runtime execution planning.

Resource Affinity is authoritative Runtime state and SHALL NOT be constructed
or forged by a Component through process-local identifiers.

A portable Component may legitimately express semantic intent such as:

- preserve the source resource's affinity
- allow the Runtime to choose a new compatible placement
- require host-accessible data
- forbid host staging
- permit explicit host staging

but it SHALL NOT express a concrete Provider or Device identity.

The current WIT world also declares:

```wit
world compute {
    export run;
}
```

while Magnetar Components conceptually import `magnetar:compute/run`.

The direction of the Component-facing contract must therefore be made explicit
and unambiguous.

Finally, Runtime-generated Provider and Device identity must be distinguished
from Component-provided routing input.

Provider and Device identifiers may legitimately appear in:

- diagnostics
- observability
- resolved execution plans
- Runtime-native bindings
- administrative inspection

but SHALL NOT be accepted as portable routing instructions from a Component.

## What Changes

This change introduces `magnetar:compute/run@2.0.0`.

The major version increase is required because the portable WIT data-movement
record changes incompatibly.

### Remove Concrete Target Selection

The following fields SHALL be removed from the portable
`data-movement-descriptor`:

```text
target-provider
target-device
target-affinity-group
```

Portable Components SHALL NOT provide:

- ProviderId
- ProviderBinding
- DeviceId
- DeviceBinding
- AffinityGroupId
- native queue identity
- native stream identity
- allocator identity
- hardware-native placement handles

as Compute routing input.

### Introduce Portable Placement Intent

The data-movement contract SHALL express portable semantic placement intent.

The initial placement intents SHALL be equivalent to:

```wit
enum placement-intent {
    preserve-source-affinity,
    runtime-selected,
    host-accessible,
}
```

The semantics are:

#### `preserve-source-affinity`

The resulting resource remains constrained by the source resource's existing
Resource Affinity unless the requested operation semantically requires a new
resource and the Runtime determines that preserving the complete affinity is
impossible.

This intent is appropriate for operations such as:

- copy within compatible placement
- materialization
- dtype conversion that remains local
- view materialization

It SHALL NOT authorize migration.

#### `runtime-selected`

The Component explicitly allows the Runtime to determine a compatible
destination according to:

- Capability requirements
- Provider advertisements
- Resource Affinity
- Resolution Policy
- Device availability
- execution planning
- memory planning
- future pressure/readiness constraints

The Component does not learn or choose the target through the request.

This intent MAY be used for explicit transfer or placement conversion.

#### `host-accessible`

The resulting representation must be consumable through a portable
host-accessible data path.

This does not select a CPU Provider.

It describes a portability property of the resulting data.

The Runtime still determines how that property is implemented.

### Host Staging Policy

The existing boolean `allow-host-staging` SHALL be replaced by explicit semantic
policy.

The initial contract SHALL distinguish at least:

```wit
enum host-staging-policy {
    forbid,
    permit,
}
```

`permit` means that host staging is semantically acceptable to the Component.

It SHALL NOT grant authority to perform host staging if Runtime policy,
Resource Affinity, Provider support, memory planning, security policy, or other
constraints prohibit it.

`forbid` means the Runtime SHALL reject a plan requiring host staging.

The portable contract SHALL NOT initially provide `force-host-staging`.

### Revised Data Movement Shape

The Component-facing descriptor SHOULD become equivalent to:

```wit
record data-movement-descriptor {
    kind: data-movement-kind,
    source: data-movement-source,
    output: tensor-descriptor,
    placement: placement-intent,
    host-staging: host-staging-policy,
}
```

The exact generated Rust representation is implementation-defined.

The portable semantics defined by this proposal are normative.

### Runtime Resolution

Portable placement intent SHALL be translated into Runtime-owned execution
constraints.

The Runtime MAY internally derive a resolved movement plan containing concepts
such as:

```text
ResolvedDataMovementPlan
├── source Resource Affinity
├── selected ProviderBinding
├── selected DeviceBinding
├── selected CapabilityBinding
├── selected memory placement
├── transfer/materialization steps
└── host staging decision
```

These resolved bindings SHALL remain internal.

They SHALL NOT become part of the Component-facing WIT request.

### Resource Affinity

A Component SHALL NOT create Resource Affinity by supplying Runtime identifiers.

Affinity is established by the Runtime as resources are created or bound.

When an existing tensor is supplied as a data-movement source, the Runtime SHALL
derive its actual affinity from Runtime-owned resource state.

The Component may request preservation of that affinity but SHALL NOT forge,
replace, or weaken it.

### Explicit Data Movement Remains Required

Removing Provider and Device target fields SHALL NOT reintroduce implicit data
movement.

Cross-placement execution still requires an explicit:

- transfer
- copy
- materialization
- upload
- download
- placement conversion

or another future explicitly defined movement operation.

The change is:

```text
explicit operation
+ portable placement intent
+ Runtime resolution
```

not:

```text
implicit Runtime migration
```

### Provider and Device Diagnostics

Provider and Device identifiers MAY remain present in Runtime-produced
diagnostics.

For example:

```wit
record compute-diagnostic {
    provider-id: option<string>,
    device-id: option<string>,
    ...
}
```

is permitted because these fields describe what the Runtime resolved or what
failed.

They are output metadata.

They SHALL NOT become input routing controls.

### Provider-Specific Extension Values

Existing Provider-specific Compute extension concepts, where retained, SHALL
not become routing mechanisms.

A Provider-specific dtype, layout, or operation extension MAY only be used
under an explicitly non-portable extension contract and after compatibility
with the selected Provider has been established.

Such values SHALL NOT cause the Runtime to interpret an embedded Provider name
as a portable Provider-selection request.

A future change MAY further separate Provider-extension schemas from the core
portable Compute WIT.

That larger separation is not required by this proposal.

### Component WIT Direction

The WIT representation SHALL explicitly describe Components as consumers of the
Compute Capability.

The previous:

```wit
world compute {
    export run;
}
```

SHALL no longer be the canonical Component-facing world.

A reference consumer world MAY instead be defined as:

```wit
world compute-consumer {
    import run;
}
```

Individual Magnetar Components MAY define their own worlds and import:

```text
magnetar:compute/run@2.0.0
```

alongside other Capabilities.

The native Provider boundary SHALL NOT be modeled as a WASM Component world.

Providers remain native Runtime extensions.

### Versioning

Because the WIT record shape changes incompatibly, the executable Compute
Capability SHALL advance from:

```text
magnetar:compute/run@1.1.0
```

to:

```text
magnetar:compute/run@2.0.0
```

Providers supporting the revised contract SHALL advertise version `2.0.0`.

A Provider advertising only `1.1.0` SHALL NOT automatically satisfy a `2.0.0`
request.

The Runtime SHALL NOT silently reinterpret the removed concrete target fields.

A temporary compatibility adapter MAY be developed separately if migration
requires one, but it SHALL NOT be part of the canonical v2 contract.

### Runtime and Native API

Runtime-native administrative policy MAY contain concrete Provider or Device
constraints where architecture requires them.

Examples may include:

- deployment policy
- debugging policy
- tests
- administrative pinning
- future cluster integration
- Runtime-created Resource Affinity

Such constraints SHALL enter resolution through native Runtime policy or
Runtime-owned bindings.

They SHALL NOT leak into the portable Compute WIT.

## Non-Goals

This change does not:

- redesign Resource Affinity
- redesign Resolution Policy
- introduce automatic migration
- introduce automatic failover for live resources
- redesign Provider advertisements
- stabilize Provider ABI
- redesign the Compute Operation Catalog
- introduce model placement
- introduce multi-device topology
- introduce distributed placement
- give Components administrative Provider selection
- define accelerator-specific placement classes
- implement inference scheduling
- implement model residency
- implement the real WASM Component engine

## Impact

The Compute WIT receives a major version increment.

Portable Components using `magnetar:compute/run@1.1.0` must migrate to the
`2.0.0` contract.

Providers must advertise the corresponding revised Compute Capability version.

The Runtime becomes the exclusive authority for mapping portable placement
intent to concrete Provider and Device bindings.

The Component boundary becomes consistent with the canonical Magnetar
architecture:

```text
Component
    |
    | "transfer this resource;
    |  runtime may select placement"
    v
Runtime
    |
    | affinity + policy + advertisements + planning
    v
Resolved Provider / Device
```

rather than:

```text
Component
    |
    | "send this to Provider X / Device Y"
    v
Runtime
```

This creates the correct foundation for future Model Components, inference
sessions, KV caches, adapters, and Tachyon-distributed Magnetar Components.