## ADDED Requirements

### Requirement: Provider Loading Modes

Magnetar SHALL distinguish Provider loading modes.

Supported modes MAY include:

- built-in
- dynamic-library
- test-provider
- development-provider

All modes SHALL register through the Runtime Provider Registry.

#### Scenario: Built-in Provider

Given a Provider is compiled into the Runtime binary

When Runtime initializes

Then it may register through the same Provider Registry as other Providers

Without defining the dynamic library ABI.

---

### Requirement: Dynamic Provider ABI Is Explicit

Dynamic Provider libraries SHALL use an explicit, versioned native ABI.

The stable dynamic ABI SHALL NOT be an implicit Rust trait-object boundary.

#### Scenario: Load dynamic Provider

Given a native library is discovered

When Runtime loads it

Then Runtime performs ABI negotiation through a stable descriptor

And does not accept a Rust `Box<dyn Provider>` as the stable compatibility
contract.

---

### Requirement: Rust Trait Objects Are In-Process Only

Rust trait objects SHALL remain limited to built-in Providers, mocks, or internal
adapters compiled together with the Runtime.

Rust trait objects SHALL NOT be the stable cross-dynamic-library Provider ABI.

#### Scenario: Mock Provider test

Given a unit test uses an in-process Rust mock implementing `Provider`

When the test runs

Then this does not imply that dynamic libraries may return `dyn Provider` as
their stable ABI.

---

### Requirement: Provider Factory Symbol

A dynamic Provider library SHALL expose a canonical factory or descriptor
symbol.

The symbol SHALL allow the Runtime to obtain ABI version and descriptor
information before registration.

#### Scenario: Missing factory symbol

Given a dynamic library lacks the required Provider factory symbol

When Runtime attempts to load it

Then loading fails before Provider registration.

---

### Requirement: Provider ABI Version

Provider dynamic ABI SHALL have an explicit ABI version.

Runtime SHALL reject unsupported ABI major versions.

#### Scenario: Unsupported ABI major

Given a Provider library reports ABI major version 99

When Runtime supports only ABI major version 1

Then the Provider is rejected.

---

### Requirement: Provider ABI Descriptor

A dynamic Provider SHALL expose a descriptor containing required ABI functions
and metadata accessors.

The descriptor SHALL be validated before registration.

#### Scenario: Missing required function

Given a Provider descriptor lacks a required status function

When Runtime validates the descriptor

Then loading fails before Provider registration.

---

### Requirement: Provider Loading Handshake

Runtime SHALL complete a loading handshake before registering a dynamic
Provider.

The handshake SHALL validate ABI, metadata, Capability advertisements, Device
metadata, status reporting, and execution API availability.

#### Scenario: Invalid advertisement

Given a Provider reports malformed Capability advertisements

When the loading handshake runs

Then the Provider is rejected before it becomes eligible for Resolution.

---

### Requirement: Provider Metadata Before Registration

Provider metadata SHALL be retrieved and validated before Provider
registration.

#### Scenario: Duplicate ProviderId

Given a dynamic Provider reports a ProviderId already registered

When Runtime validates metadata

Then loading fails or follows explicit duplicate policy.

---

### Requirement: Provider Capability Advertisements Through ABI

Dynamic Providers SHALL expose Capability advertisements through the Provider
ABI.

Advertisements SHALL be validated before Provider eligibility.

#### Scenario: Advertise Compute

Given a Provider advertises Compute Capability support

When Runtime validates the advertisement

Then malformed or incompatible advertisements are rejected.

---

### Requirement: Provider Device Metadata Through ABI

Dynamic Providers SHALL expose Device metadata through the Provider ABI.

Device metadata SHALL not expose raw native handles as public Runtime API.

#### Scenario: List GPU Device

Given a Provider exposes a GPU Device

When Runtime reads Device metadata

Then Runtime receives stable Device metadata

And not a raw CUDA, HIP, Metal, or driver handle.

---

### Requirement: Provider Status Through ABI

Dynamic Providers SHALL expose status through the ABI using the refined status
model.

Status SHALL include or support lifecycle, health, readiness, pressure,
admission, freshness, Device status, and Capability status.

#### Scenario: Provider saturated

Given a dynamic Provider reports saturated pressure

When Runtime reads status through the ABI

Then Runtime can distinguish saturation from Provider failure.

---

### Requirement: Provider Execution Through ABI

Dynamic Providers SHALL expose execution behavior through a stable ABI-compatible
execution surface.

The ABI SHALL preserve ProviderExecutionApi semantics without exposing arbitrary
Rust types.

#### Scenario: Submit execution

Given Runtime submits Provider execution through dynamic ABI

When the Provider accepts work

Then request and response payloads follow ABI-compatible structures

And not private Rust layouts.

---

### Requirement: ABI Memory Ownership

All memory crossing the Provider ABI boundary SHALL have explicit ownership.

#### Scenario: Provider returns string

Given Provider returns a diagnostic string

When Runtime consumes it

Then the ABI defines whether Runtime must call a Provider release function

And Runtime does not free the memory with the wrong allocator.

---

### Requirement: Provider Opaque Handles

Dynamic Provider state SHALL use opaque handles only as Runtime-internal native
state handles.

Opaque handles SHALL have explicit destroy or release functions.

Opaque handles SHALL not be exposed through Component WIT or public portable
APIs.

#### Scenario: Provider-owned tensor handle

Given Provider returns an opaque native resource handle

When Runtime stores it internally

Then the handle remains Runtime/Provider internal

And is not exposed to portable Components.

---

### Requirement: No Unwind Across Provider ABI

Provider calls SHALL NOT unwind across the ABI boundary.

#### Scenario: Provider panics

Given a Rust-based Provider panics internally

When execution crosses the ABI boundary

Then the Provider adapter catches or handles the panic according to policy

And Runtime receives a stable failure or marks the Provider failed.

---

### Requirement: Provider ABI Error Model

Provider ABI calls SHALL report stable error categories that Runtime can
normalize.

#### Scenario: Provider not ready

Given Provider rejects work because it is not ready

When the error crosses the ABI

Then Runtime maps it to a Provider-not-ready style error

And not to an opaque native failure.

---

### Requirement: Provider Threading Declaration

A dynamic Provider SHALL declare threading and reentrancy expectations.

Runtime SHALL respect the declaration.

#### Scenario: Single-threaded Provider

Given a Provider declares single-threaded execution

When Runtime schedules calls into it

Then Runtime serializes access or rejects the Provider according to policy.

---

### Requirement: Provider Blocking Behavior Declaration

A dynamic Provider SHALL declare relevant blocking or asynchronous execution
behavior.

#### Scenario: Long-running blocking Provider

Given a Provider declares that execution calls may block

When Runtime uses the Provider

Then Runtime isolates or schedules those calls according to Runtime policy.

---

### Requirement: Provider Library Unloading Safety

Runtime SHALL not unload a dynamic Provider library while Provider code may
still be referenced.

#### Scenario: In-flight operation

Given a Provider has an in-flight operation

When Runtime stops the Provider

Then Runtime does not unload the library until unloading is safe

Or it follows a conservative never-unload policy.

---

### Requirement: Provider Loading Is Trusted Native Code

Dynamic Provider loading SHALL be treated as trusted native code execution.

The ABI boundary is not a sandbox.

#### Scenario: Configure Provider path

Given Runtime is configured to load a Provider library

When policy evaluates the path

Then only allowed paths or trusted Provider packages are loaded.
