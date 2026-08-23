## ADDED Requirements

### Requirement: Runtime Rejects Trait-Object Dynamic Providers

Runtime SHALL NOT treat dynamic libraries returning Rust trait objects as the
stable Provider loading ABI.

#### Scenario: Trait object factory

Given a dynamic library exposes a factory returning `Box<dyn Provider>`

When Runtime applies the stable dynamic Provider loading policy

Then the library is rejected or handled only by a non-stable development
compatibility path.

---

### Requirement: Runtime Performs Provider ABI Handshake

Runtime SHALL perform a Provider ABI handshake before registering a dynamic
Provider.

#### Scenario: Handshake succeeds

Given a dynamic library exposes a supported ABI descriptor

And metadata, advertisements, Devices, status, and execution functions validate

When Runtime completes handshake

Then the Provider may be registered.

---

### Requirement: Runtime Keeps Provider ABI Internal

Provider ABI descriptors, opaque handles, and native function tables SHALL remain
Runtime-internal.

They SHALL not be exposed to Components, WIT, or public portable APIs.

#### Scenario: Component invokes Compute

Given a Component invokes Compute

When Runtime resolves a dynamic Provider

Then the Component interacts only with the portable Capability contract

And does not see ABI handles.

---

### Requirement: Runtime Validates ABI Version

Runtime SHALL reject unsupported Provider ABI versions before Provider
registration.

#### Scenario: ABI version mismatch

Given a Provider reports unsupported ABI version

When Runtime loads it

Then Runtime rejects it with a structured loading error.

---

### Requirement: Runtime Validates Provider Metadata

Runtime SHALL validate Provider metadata during loading.

#### Scenario: Invalid ProviderId

Given Provider metadata contains an invalid ProviderId

When Runtime performs loading handshake

Then Runtime rejects the Provider.

---

### Requirement: Runtime Validates Provider Advertisements

Runtime SHALL validate Capability advertisements during loading.

#### Scenario: Malformed Capability version

Given a Provider advertisement contains malformed Capability version

When Runtime loads the Provider

Then registration fails.

---

### Requirement: Runtime Validates Provider Devices

Runtime SHALL validate Device metadata during loading.

#### Scenario: Duplicate DeviceId

Given a Provider reports duplicate Device identities

When Runtime validates Devices

Then loading fails or follows explicit duplicate Device policy.

---

### Requirement: Runtime Applies Loading Policy

Runtime SHALL apply loading policy before executing dynamic Provider code beyond
the required safe handshake.

Policy MAY include allowed paths, trusted digests, signatures, development mode,
and revocation.

#### Scenario: Disallowed Provider path

Given a Provider library path is outside configured allowed locations

When Runtime attempts loading

Then loading is denied by policy.

---

### Requirement: Runtime Normalizes Provider Loading Errors

Runtime SHALL normalize Provider loading and ABI errors into stable Runtime
errors.

#### Scenario: Descriptor invalid

Given ABI descriptor validation fails

When Runtime reports the error

Then callers receive a stable Provider loading error category.

---

### Requirement: Runtime Respects Provider Threading Model

Runtime SHALL respect the Provider's declared threading and reentrancy model.

#### Scenario: Runtime-synchronized Provider

Given a Provider declares it requires Runtime synchronization

When Runtime calls it

Then Runtime serializes calls according to that declaration.

---

### Requirement: Runtime Respects Provider Blocking Declaration

Runtime SHALL treat blocking Provider calls according to Runtime execution
policy.

#### Scenario: Blocking execution call

Given a Provider declares blocking execution behavior

When Runtime schedules execution

Then Runtime avoids blocking critical Runtime control paths.

---

### Requirement: Runtime Prevents Unsafe Library Unload

Runtime SHALL not unload a Provider library while Provider resources,
operations, callbacks, or threads may still reference it.

#### Scenario: Provider resource exists

Given a Provider-owned tensor resource still exists

When Runtime stops the Provider

Then the library remains loaded or resource destruction occurs before unloading.

---

### Requirement: Runtime Observes Provider Loading

Runtime SHALL emit observability events for Provider loading lifecycle and
failures.

#### Scenario: Factory symbol missing

Given a Provider library lacks the expected factory symbol

When loading fails

Then Runtime may emit an observation with a stable failure reason.

---

### Requirement: Runtime Treats Dynamic Providers As Trusted Native Code

Runtime SHALL document and enforce that dynamically loaded Providers are trusted
native code.

Runtime SHALL not describe the ABI boundary as a security sandbox.

#### Scenario: Untrusted Provider library

Given an untrusted Provider binary is available

When Runtime policy evaluates it

Then Runtime refuses to load it unless policy explicitly trusts it.
