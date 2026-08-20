# Compute Capability design

## Scope

Compute is a portable, provider-backed capability. This change defines its
identity and contract boundary only; it does not standardize tensor, kernel, or
other mathematical operations.

## Contract

The capability identifier and its sole WIT interface are both
`magnetar:compute/run`. Version 1.0.0 is represented by the WIT package
`magnetar:compute@1.0.0` and an empty `run` interface. The empty interface is a
deliberate capability marker: operations can be added in a later compatible
minor release after their semantics are designed.

The canonical WIT source lives at `wit/compute.wit`. Runtime code exposes the
same values through `compute_capability()`, so providers advertise the exact
contract instead of duplicating string literals.

## Registration and resolution

Providers add `compute_capability()` to `ProviderMetadata.capabilities`. During
registration, the existing capability registry validates and indexes the
contract. Components request the WIT interface; the registry resolves a
provider with a compatible semantic capability version. The selected provider
is never exposed to the component.

## Versioning and lifecycle

The capability version uses the runtime's semantic-version compatibility rules:
for stable major versions, an available version satisfies an equal-or-earlier
requirement with the same major version. A breaking contract change requires a
new major version. A provider may be registered or removed independently; when
no compatible provider remains, resolution returns no provider.
