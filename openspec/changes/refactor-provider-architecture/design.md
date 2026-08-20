# Provider and capability resolution design

## Scope

This change is a terminology and contract refactor of the existing runtime.
It preserves backend registration and selection while making the native
extension boundary capability-oriented. It does not add hardware providers or
execute work through a capability yet.

## Model

- A `Provider` is a native extension with `ProviderMetadata`.
- Metadata contains the provider API version and a set of `Capability` values.
- A capability is identified by its WIT package-qualified name and an
  independently versioned string.
- `ProviderRegistry` owns registered backends and indexes each capability to
  the providers that advertise it.
- `ProviderLoader` owns provider lifecycle and dynamic libraries. It registers
  a provider atomically: an error from registration or initialization restores
  the registry and leaves the failing provider unavailable.

## Resolution and fallback

Capability versions are exact WIT contract versions in this change. A request
therefore resolves only providers that advertise the same name and version.
When several providers match, the registry returns them in deterministic
provider-name order; callers can try each in turn, so the next compatible
provider is the fallback. Semver range negotiation and execution health checks
are deferred until a scheduler exists.

Components keep declaring WIT imports and never hold a provider reference.
The runtime resolves a provider only from such a capability request.

## Compatibility

The rename is intentionally breaking because this pre-1.0 crate exposes the
extension contracts publicly. Dynamic libraries export
`magnetar_provider_create` and must target `PROVIDER_API_VERSION`.

## Validation

Unit tests cover successful and failed provider registration, capability
registration and deterministic fallback resolution. Existing component and
backend lifecycle tests continue to cover unchanged runtime behavior.
