# Resource Affinity Model Design

## Context

Magnetar currently registers Providers, Devices, and versioned Capabilities and
can enumerate compatible Providers. The resolver has no execution boundary and
no host representation for WIT resources, so the original change could not
truthfully attach affinity to concrete tensor, model, tokenizer, or generation
objects. Only tensor, graph, and operation resource declarations exist, in the
`magnetar:compute@1.1.0` WIT contract; model-level contracts remain provisional
taxonomy entries.

This design establishes the runtime-native foundation needed by future host
adapters. It keeps WIT unchanged and adds an affinity-aware resolver that can be
tested against today's Provider registry.

## Goals / Non-Goals

**Goals:**

- Represent immutable ownership and compatibility facts with stable Rust data
  types.
- Aggregate all live-resource constraints before selecting one Provider.
- Preserve exact versions for live resources while retaining semantic version
  negotiation for initial resolution.
- Return structured, dimension-specific incompatibility errors.
- Provide a generic host envelope for current and future opaque resources.
- Make fallback phase-aware and prohibit implicit migration of live state.

**Non-Goals:**

- Change `magnetar:compute@1.1.0` or expose affinity through WIT.
- Invent host bindings for a WASM engine that is not yet integrated.
- Define model loading, tokenization, prompt-template, or generation contracts.
- Transfer, migrate, serialize, or recover live Provider-owned resources.
- Change the `Provider` trait, provider ABI version, or existing stateless
  resolution APIs.

## Decisions

### Keep affinity host-side and additive

Affinity types live in `magnetar-runtime` and existing APIs remain available.
`AffinityResource<T>` carries a native value and immutable affinity, while the
affinity itself contains no native handle. Future host adapters wrap generated
WIT resource representations at the point where the Runtime has authoritative
selection data.

Alternative: expose affinity from Compute WIT resources. Rejected because that
would require a new WIT version and leak scheduling identity to Components
before a portable projection has been designed.

### Separate resource facts from aggregate constraints

`ResourceAffinity` describes one resource. `AffinityConstraints` combines all
resource facts for one dependent resolution and reports a conflict instead of
discarding a binding. This keeps the input immutable and prevents validation
from being confused with resource mutation.

Alternative: merge by overwriting optional fields on one descriptor. Rejected
because the last resource would silently win on incompatible bindings.

### Treat bindings as exact after resource creation

Provider and Device identifiers, execution-context and group identifiers,
Capability versions, and same-role artifact fingerprints must match exactly
when both sides constrain them. Distinct Capability IDs and artifact roles are
retained because one chain may contain several capabilities and artifacts.

Semantic compatibility remains the rule for selecting a Capability before a
new resource exists. If a live resource already records the requested
Capability ID, its exact version is used.

Alternative: apply semantic-version compatibility between live handles.
Rejected because an opaque resource type or Provider implementation cannot be
assumed interchangeable across WIT package versions.

### Use named artifact roles and an explicit shared bundle role

`ArtifactBinding` pairs a role with a canonical fingerprint. Aggregation only
compares bindings with the same role. Model, tokenizer, and template resources
can each retain their own digest and also declare a shared `model-bundle` or
`compatibility-manifest` fingerprint.

Alternative: compare every artifact fingerprint for equality. Rejected because
different files normally have different content digests even when they belong
to one compatible bundle.

### Resolve the best compatible version inside an existing Provider binding

Affinity-aware resolution first aggregates Provider and Device constraints.
When a Provider is bound, the registry searches compatible versions advertised
by that Provider. It does not first choose the globally newest version and
then filter, which could incorrectly exclude a bound Provider that supports a
slightly older compatible version.

The resolver returns one `AffinityResolution`: Provider, exact Capability, and
the affinity facts to attach to the next resource. Stateless callers can keep
using the existing multi-Provider resolver.

Alternative: filter the output of the existing resolver. Rejected because its
global version choice happens before Provider filtering.

### Make contexts and groups runtime-local identities

Every built Runtime receives a monotonic process-local `ExecutionContextId`.
Affinity-aware resolution rejects resources from another context. A dependent
resolution preserves its input group or creates an `AffinityGroupId` when no
input is grouped. Independent resources remain ungrouped and shareable; shared
inputs are not mutated to join every session group.

Alternative: use globally persistent UUIDs. Rejected because live resource
migration and persistence are explicitly out of scope and no new dependency is
needed for process-local safety.

### Treat fallback classification as recovery information

The ordering is transparent < restartable < Provider-pinned, and aggregation
keeps the most restrictive value. A Provider or Device binding remains
authoritative for a live call regardless of classification. Restartable means
the caller may explicitly recreate state from replayable input; it never means
the Runtime can substitute another Provider for an existing handle.

Alternative: let restartable resolution ignore a missing Provider binding.
Rejected because that would pass a live opaque handle to an implementation that
does not own it.

## Risks / Trade-offs

- **Provider names are runtime-local identities, not persistent instance
  UUIDs** -> Provider registration already enforces unique names and affinities
  are not valid across Runtime lifetimes.
- **Generic envelopes do not automatically wrap WIT resources** -> Each future
  host adapter must attach the Runtime-produced affinity when it creates its
  native resource representation.
- **Artifact compatibility depends on a producer declaring a shared role** ->
  Model-contract changes must specify the manifest or bundle fingerprint source.
- **One coherent Provider may be too strict for future cross-Provider transfer**
  -> Explicit transfer creates a new resource and affinity; transfer semantics
  require a separate change.
- **Process-local counters are not serialization identities** -> Serialization
  and live migration remain out of scope.

## Migration Plan

Add the new APIs without changing existing Provider implementations or
stateless resolution. Future host adapters opt into `AffinityResource<T>` and
`resolve_with_affinity` when they begin creating or consuming opaque state. A
rollback removes only the additive types, resolver, tests, and documentation;
there is no persisted or live-state migration.

## Open Questions

- Which WASM Component host will own the first concrete Compute resource
  adapter?
- Which manifest format will define shared model/tokenizer/template bundle
  fingerprints?
- Which explicit transfer contract will create a resource with new Provider or
  Device affinity?
