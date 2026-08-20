# Capability model design

## Scope

This change replaces the former two-string `Capability` label with an
independently registered, versioned capability contract. It remains a runtime
metadata model: providers advertise implementations and components request
contracts; neither side gains a direct reference to the other.

## Public model

- `CapabilityId` is a non-empty, globally unique package-qualified identifier.
- `CapabilityVersion` is an ordered semantic version (`major.minor.patch`).
  Pre-release and build metadata are deliberately out of scope for the first
  runtime API.
- `CapabilityDescriptor` contains a non-empty set of WIT interfaces, an
  optional description, and dependency version requirements keyed by
  `CapabilityId`.
- `Capability` combines its identifier, version, and descriptor. A provider
  advertises a full `Capability` value in its metadata; the runtime deduplicates
  this definition by identifier and version, then records the provider as an
  implementation of it.

## Registration and validation

`ProviderRegistry` owns the capability catalog independently from its provider
index. Registering a capability validates its non-empty identifier and WIT
contract set. Registering the same identifier and version with a different
descriptor is rejected; an identical definition is shared by additional
providers.

Dependencies are deliberately validated after registration through an explicit
registry validation step. This permits definitions to be registered in any
order, while preventing execution or resolution from using a capability whose
dependencies cannot be satisfied.

## Version compatibility and resolution

Version `A` satisfies a request for `R` when both have the same non-zero major
version and `A >= R`. The registry resolves the highest such registered
version. The zero-major series is treated conservatively: only an exact version
is compatible. The provider list for the resolved version is ordered by provider
name, preserving deterministic fallback.

WIT contracts are exact values in this change. Semantic-version negotiation is
applied to the capability version, not to individual WIT interface versions.

## Lifecycle and relationships

Capabilities are registered before providers become selectable. Provider
registration remains atomic: a registration or initialization failure restores
both the capability catalog and provider index. Shutdown clears capability
definitions and provider associations with the rest of the registry.

Components continue to express requirements through WIT imports. The runtime
can translate an import into the corresponding single-contract capability and
resolve compatible providers without leaking a provider reference into the
component contract.

## Errors and compatibility

The registry reports invalid definitions, conflicting definitions, missing or
incompatible dependencies, and unresolved capability requests through
`ProviderError`. This is intentionally a pre-1.0 breaking API change: callers
must construct semantic versions and descriptors instead of supplying arbitrary
version strings.
