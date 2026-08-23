# Provider Loading and ABI Policy

Magnetar treats dynamically loaded Providers as trusted native code. The native
Provider ABI is an internal Runtime boundary, not a Component ABI, not WIT, and
not a sandbox.

## Loading modes

The Runtime distinguishes four loading modes:

- `BuiltIn`: Provider is compiled into the Runtime process and may use Rust
  traits internally.
- `DynamicLibrary`: Provider is discovered as a native shared library and must
  pass loading policy and ABI validation.
- `TestProvider`: in-process test double used by unit and integration tests.
- `DevelopmentProvider`: explicit local development mode for native libraries;
  it may relax path trust, but ABI validation still runs.

All usable Providers register through `ProviderRegistry`. Built-in and test
Providers may use `Arc<dyn Provider>` inside the same binary. That Rust trait
object shape is not the stable dynamic-library contract.

## Dynamic ABI

The stable dynamic ABI is versioned by `ProviderAbiVersion`. Runtime support is
currently `1.0`, exposed through `PROVIDER_ABI_FACTORY_SYMBOL_V1` as
`magnetar_provider_v1`.

Dynamic libraries must expose a stable descriptor-oriented ABI. The old
prototype shape `magnetar_provider_create -> Box<dyn Provider>` is rejected as
a stable cross-library contract because Rust does not stabilize trait object
layout, vtables, allocator ownership, panic behavior, or compiler compatibility
across dynamic library boundaries.

`ProviderAbiDescriptor` defines the required shape:

- descriptor size/layout guard
- ABI version
- required metadata, advertisement, device, status, execution, release, and
  destroy functions
- optional features declared by flags
- ownership rules
- threading model
- blocking or async behavior
- unload policy

The Runtime validates a descriptor before Provider registration. Unsupported ABI
major versions are rejected. Minor versions are accepted only when they do not
exceed the Runtime-supported minor version.

## Handshake

Dynamic loading follows this sequence:

1. discover library
2. apply loading policy
3. load library
4. resolve `magnetar_provider_v1`
5. retrieve descriptor
6. validate descriptor and ABI version
7. retrieve and validate Provider metadata
8. retrieve and validate Capability advertisements
9. retrieve and validate Device metadata
10. retrieve initial status
11. initialize Provider
12. register Provider through `ProviderRegistry`
13. mark Provider ready

Registration happens only after the handshake succeeds.

## Metadata, advertisements, devices and status

Provider metadata includes Provider identity, version, vendor, description,
Runtime compatibility, feature flags and loading mode. Capability
advertisements include Capability identifiers and versions plus compute,
operation-family and data-movement support where implemented. Device metadata
includes stable Device identity, type, Provider ownership and available memory
or feature metadata.

Native device handles are not public Runtime API. Provider status uses the
Runtime status model: lifecycle, health, readiness, pressure, admission,
freshness, Device status, Capability status and redacted diagnostics.

## Execution

Dynamic execution uses ABI-compatible request and response payloads. The ABI
must preserve `ProviderExecutionApi` semantics, including validation, Resource
Affinity, Provider-owned resources, Device-bound resources, cancellation,
structured errors and observability correlation. Arbitrary Rust request or
response types are not ABI payloads.

## Memory ownership

Memory crossing the ABI boundary has explicit ownership. Provider-owned strings,
lists, descriptors, error messages, opaque handles and buffers require
Provider-side release or destroy functions. Runtime-owned buffers are borrowed
for the call unless ownership is explicitly transferred.

The Runtime must not free Provider-allocated memory with the Runtime allocator.
The Provider must not free Runtime-owned memory unless ownership was explicitly
transferred.

## Opaque handles

Opaque handles may represent Provider instances, Provider-owned resources or
operations. They remain Runtime/Provider internals, are not WIT handles, are not
Component handles and are not serialized as stable public identifiers. Every
handle kind requires an explicit destroy or release path.

## Panic and unwind boundary

Provider ABI calls must not unwind across the ABI boundary. Rust-based Provider
adapters catch panics or abort according to policy. A panic or unwind boundary
violation is treated as Provider failure and normalized into a stable Runtime
error.

## Threading and blocking

A dynamic Provider declares whether it is single-threaded,
Runtime-synchronized, internally thread-safe or reentrant. The Runtime respects
that declaration. A Provider also declares blocking, async-capable or
long-running execution behavior so the Runtime can schedule, isolate or cancel
work without blocking critical control paths.

## Unloading

The conservative policy is `NeverUnload`. If unloading is implemented later, the
Runtime must prove that no Provider instances, Provider-owned resources,
in-flight operations, callbacks or background threads can still reference code
from the library.

## Security

Dynamic Providers are trusted native code execution in the Runtime process. The
loading policy must gate arbitrary paths by default and may additionally use
digest allowlists, signature metadata, publisher trust and revocation. Provider
metadata cannot grant trust by itself.

Development mode is explicit and local. It may permit local unsigned libraries,
but descriptor and ABI validation still run.
