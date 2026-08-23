# Define Provider Loading and ABI Policy

## Why

Magnetar Providers are trusted native extensions.

They implement Capabilities, expose Devices, report health/readiness/pressure,
and execute inference-related work through Runtime-owned policy.

The current dynamic Provider loading approach is suitable for early prototyping,
but it must not become the stable native extension ABI.

A dynamic library returning a Rust trait object such as:

```rust
Box<dyn Provider>
```

across a dynamic library boundary is not a stable ABI contract.

Rust does not provide a stable cross-dynamic-library ABI for trait objects,
vtable layout, allocation ownership, panic behavior, or compiler-version
compatibility.

If Magnetar stabilizes that shape, Provider compatibility would depend on
unwritten assumptions such as:

- same Rust compiler version
- same crate versions
- same trait object layout
- same allocator expectations
- same panic strategy
- same build flags
- same feature flags
- same dependency graph
- same private type layout

That is not acceptable for a long-term Provider extension mechanism.

Magnetar needs a clear Provider loading and ABI policy before Provider
conformance and native extension distribution are stabilized.

This change defines:

- supported Provider loading modes
- stable native ABI expectations
- factory symbol policy
- ABI version negotiation
- ownership and destruction rules
- allocator boundaries
- panic boundaries
- error reporting
- metadata discovery
- capability advertisement retrieval
- health/readiness/pressure reporting
- Provider execution API binding
- compatibility checks
- unsafe boundary requirements
- test requirements

This change does not implement the full ABI.

It defines the policy that future implementation must satisfy.

## What Changes

Magnetar SHALL define Provider loading as a trusted native extension boundary.

The canonical Provider loading model SHALL support at least:

- statically linked built-in Providers
- dynamically loaded native Providers

Both modes SHALL be represented behind the same Runtime Provider Registry.

Dynamic loading SHALL use a stable ABI boundary.

### Provider Loading Modes

Magnetar MAY support several Provider loading modes:

```text
built-in
dynamic-library
test-provider
development-provider
```

A built-in Provider is compiled into the Runtime or linked as part of the same
binary.

A dynamic-library Provider is loaded at runtime from a native shared library.

A test-provider is used only by tests.

A development-provider may use relaxed local loading policy but SHALL still not
define the long-term stable ABI by accident.

### Rust Trait Objects Are Not Stable ABI

The stable dynamic Provider boundary SHALL NOT require a dynamic library to
return Rust trait objects across the library boundary.

The following patterns SHALL NOT be the stable Provider ABI:

```rust
extern "Rust" fn create() -> Box<dyn Provider>
extern "C" fn create() -> *mut dyn Provider
extern "Rust" fn create() -> Arc<dyn Provider>
```

Rust trait objects MAY remain internal to one compiled binary or to one adapter
crate compiled together with the Runtime.

They SHALL NOT be the cross-library compatibility contract.

### Stable ABI Strategy

The stable dynamic Provider ABI SHOULD use a C-compatible ABI or another
explicitly versioned stable ABI.

The initial policy SHOULD prefer a C-compatible ABI because it makes the
boundary explicit.

A dynamic Provider library SHOULD expose a factory symbol such as:

```text
magnetar_provider_v1
```

or equivalent.

The symbol SHALL return a stable ABI descriptor, not a Rust trait object.

Conceptually:

```text
dynamic library
    |
    | magnetar_provider_v1()
    v
ProviderAbiDescriptor
    |
    +-- abi version
    +-- provider metadata function
    +-- capability advertisement function
    +-- device listing function
    +-- status function
    +-- execution function table
    +-- destroy / release functions
```

Exact C types and function signatures are deferred to implementation.

The semantic requirements are defined by this change.

### ABI Version

The Provider ABI SHALL have an explicit ABI version.

The Runtime SHALL reject Providers with unsupported ABI major versions.

Minor ABI compatibility MAY be allowed if policy defines backward-compatible
extension rules.

ABI version SHALL be separate from:

- Provider version
- Capability version
- Runtime version
- Component WIT version
- model artifact version

### Handshake

Provider loading SHALL perform a handshake before registration.

The handshake SHALL validate:

- expected factory symbol exists
- ABI version is supported
- ABI descriptor is well-formed
- required function pointers are present
- Provider metadata is valid
- Provider identity is valid
- Provider version is valid
- Capability advertisements are valid
- Device metadata is valid
- status reporting contract is available
- execution API contract is available for advertised executable Capabilities
- required Runtime features are supported
- optional Provider features are negotiated

A Provider SHALL NOT be registered as ready until handshake succeeds.

### Metadata First

Provider metadata SHALL be retrievable before executable work is accepted.

Metadata SHOULD include:

- ProviderId
- Provider name
- Provider version
- vendor
- description
- supported ABI version
- supported Runtime compatibility
- supported features
- loading mode
- build/provenance metadata where available

Metadata retrieval SHALL be safe to call during loading.

### Capability Advertisement

A dynamic Provider SHALL advertise supported Capabilities through the ABI.

Advertisements SHALL be validated before registration.

The Runtime SHALL reject a Provider whose advertised Capability metadata is
malformed or incompatible.

Advertisements SHALL not grant execution authority by themselves.

The Runtime still applies policy, readiness, health, Resource Affinity, and
Resolution Policy.

### Device Listing

A dynamic Provider SHALL expose Device metadata through the ABI.

Device metadata SHALL be validated.

Device identity SHALL remain Provider-owned.

A Provider SHALL NOT expose raw native device handles as stable Runtime public
API.

### Status Reporting

The ABI SHALL expose Provider status according to the refined health,
readiness, pressure, admission, freshness, Device status, and Capability status
model.

Status reporting SHALL not rely on engine-native or Provider-private structs
whose layout is unknown to the Runtime.

### Execution API Binding

Provider execution SHALL be exposed through a stable ABI-compatible function
table or equivalent stable bridge.

The ABI SHALL cover the Provider execution semantics required by Magnetar's
ProviderExecutionApi.

Execution calls SHALL preserve:

- explicit request validation
- Resource Affinity
- Provider-owned resource semantics
- Device-bound resource semantics
- cancellation semantics
- structured error mapping
- observability correlation
- no raw native handle leakage to Components

The ABI SHALL NOT expose arbitrary Rust types as request or response payloads.

### Memory Ownership

All memory crossing the ABI boundary SHALL have explicit ownership.

The policy SHALL define:

- who allocates buffers
- who frees buffers
- whether the Runtime may retain pointers
- whether the Provider may retain pointers
- how strings are encoded
- how lists are encoded
- how opaque handles are destroyed
- how error messages are released
- how versioned descriptors are released

The Runtime SHALL NOT free Provider-allocated memory with the wrong allocator.

The Provider SHALL NOT free Runtime-owned memory unless ownership was
explicitly transferred.

### Opaque Handles

The ABI MAY use opaque handles for Provider-owned state.

Opaque handles SHALL have explicit destroy/release functions.

Opaque handles SHALL not become portable Component handles.

Opaque handles SHALL not be exposed through WIT.

Opaque handles SHALL not be serialized as stable public identifiers.

### Panic and Unwind Boundary

Provider ABI calls SHALL NOT unwind across the ABI boundary.

If Rust is used inside a Provider, the Provider adapter SHALL catch panics or
abort according to policy before unwinding crosses into the Runtime.

The Runtime SHALL treat an unwind/panic boundary violation as Provider failure.

### Error Model

ABI calls SHALL return stable error categories.

Provider-native errors MAY include redacted diagnostic text.

The ABI error model SHALL distinguish:

- invalid ABI descriptor
- unsupported ABI version
- invalid metadata
- invalid advertisement
- invalid device metadata
- initialization failure
- provider not ready
- provider draining
- provider saturated
- execution rejected
- execution failed
- cancellation unsupported
- cancellation failed
- resource invalid
- internal Provider error
- panic/unwind violation

Errors SHALL be normalized into Magnetar Runtime errors.

### Thread Safety

The ABI SHALL declare thread-safety expectations.

A Provider SHALL declare whether calls are:

- single-threaded
- externally synchronized by Runtime
- internally thread-safe
- reentrant
- async-capable

The Runtime SHALL respect the declared threading model.

Provider declarations SHALL be validated against Runtime policy.

### Async and Blocking Behavior

The ABI SHALL distinguish blocking and asynchronous execution behavior.

A Provider SHALL declare whether execution calls may block.

If a Provider performs long-running work, the Runtime SHALL be able to schedule,
cancel, or isolate it according to Runtime policy.

This change does not require one universal async ABI.

It requires the behavior to be explicit.

### Lifecycle

Dynamic Provider lifecycle SHALL be explicit:

```text
library discovered
    |
    v
library loaded
    |
    v
factory symbol resolved
    |
    v
ABI descriptor validated
    |
    v
Provider initialized
    |
    v
Provider registered
    |
    v
Provider ready
    |
    v
draining / stopped / failed
    |
    v
Provider destroyed
    |
    v
library unloaded where safe
```

A library SHALL NOT be unloaded while Provider-owned resources or in-flight
operations may still reference its code.

### Library Unloading

Unloading dynamic Provider libraries is optional.

If supported, unloading SHALL be safe.

The Runtime SHALL not unload a library while:

- Provider instances exist
- Provider-owned resources exist
- in-flight operations exist
- callbacks may still occur
- background Provider threads may still call Runtime APIs

A conservative implementation MAY never unload dynamic libraries during process
lifetime.

### Security

Providers are trusted native code.

The ABI boundary does not sandbox Providers.

Loading a Provider means running native code in the Runtime process unless a
future out-of-process Provider model is defined.

Therefore, dynamic Provider loading SHALL be governed by policy.

Policy SHOULD validate:

- allowed library paths
- trusted digests
- trusted publishers where supported
- ABI version
- Runtime compatibility
- optional signatures
- development mode
- revocation

This change defines loading policy, not full Provider binary trust and
distribution.

### Development Mode

Development mode MAY allow loading local unsigned Provider libraries.

Development mode SHALL be explicit.

Development mode SHALL still validate ABI version and descriptor structure.

Development mode SHALL not silently become production policy.

### Built-In Providers

Built-in Providers MAY continue to use Rust traits internally because they are
compiled together with the Runtime.

Built-in Rust trait use SHALL NOT define the dynamic library ABI.

The Runtime may adapt built-in Providers into the same Provider Registry through
an internal adapter.

### Test Providers

Tests MAY use Rust trait objects, mocks, or in-process fake Providers.

Tests SHALL not imply that trait object dynamic loading is stable ABI.

ABI-specific tests SHALL use ABI-shaped fixtures.

### Observability

Provider loading SHALL emit Runtime observations for:

- library discovered
- library load attempted
- factory symbol missing
- ABI version rejected
- descriptor invalid
- metadata invalid
- Provider initialized
- Provider registered
- Provider ready
- Provider loading failed
- Provider destroyed
- library unloaded where supported

Observability SHALL not leak secrets, private keys, or unsafe native handles.

### Documentation

The repository SHALL document:

- Provider loading modes
- why Rust trait objects are not stable ABI
- dynamic Provider handshake
- ABI versioning
- memory ownership
- lifecycle
- error mapping
- security assumptions
- development mode
- built-in Provider behavior
- test Provider behavior

## Non-Goals

This change does not:

- implement the complete C ABI
- define Provider binary package distribution
- define Provider signing PKI
- define out-of-process Providers
- sandbox native Providers
- define remote Providers
- define Tachyon Provider distribution
- define full GPU driver isolation
- define stable model artifact ABI
- redefine Component WIT
- make Providers portable
- make Provider ABI equivalent to Component ABI
- remove built-in Providers
- require dynamic library unloading

## Impact

Magnetar gains a clear policy for native Provider loading.

The Runtime no longer treats a Rust trait object factory as the stable dynamic
Provider interface.

Future Provider ABI implementation can proceed with clear expectations for:

- version negotiation
- stable descriptors
- metadata
- Capability advertisements
- Device metadata
- status reporting
- execution API bridging
- memory ownership
- panic safety
- error normalization

This prepares the project for Provider conformance testing and safer native
extension loading.