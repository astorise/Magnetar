## MODIFIED Requirements

### Requirement: Providers

The Runtime SHALL use Provider as its sole native extension mechanism.

Providers SHALL implement one or more Capabilities.

Providers SHALL expose zero or more Devices.

Providers SHALL remain distinct from portable Components.

The Runtime SHALL NOT require a parallel Backend abstraction.

#### Scenario: Register Provider

Given a valid Provider

When the Runtime registers the Provider

Then its Capabilities and Devices become available to Runtime resolution

And no Backend registration is required.

---

### Requirement: Capability Resolution

The Runtime SHALL resolve Providers through requested Capabilities.

Components SHALL never directly reference a Provider.

Provider selection SHALL use Resolution Policy and SHALL respect Resource
Affinity.

There SHALL NOT be a Backend selection path that bypasses Provider resolution.

#### Scenario: Resolve Provider

Given multiple Providers implement a requested Capability

When a Component requests that Capability

Then the Runtime evaluates compatible Providers through Resolution Policy

And selects an eligible Provider and Device

Without consulting a Backend registry.

---

## ADDED Requirements

### Requirement: Provider Registry Exclusivity

ProviderRegistry SHALL be the sole registry for native execution extensions.

ProviderRegistry SHALL NOT contain or delegate to a separate Backend registry.

#### Scenario: Inspect native registrations

Given CPU and CUDA native implementations are registered

When the Runtime inspects its native execution registry

Then both implementations are Providers

And no Backend collection exists.

---

### Requirement: Provider Owns Device Exposure

Devices used for native execution SHALL be exposed through Providers.

A Device SHALL identify its owning Provider where ownership is required for
resolution and Resource Affinity.

#### Scenario: Discover GPU Device

Given a CUDA Provider discovers a GPU

When the Runtime receives the Device metadata

Then the Device is associated with the CUDA Provider

And no Backend owner is required.

---

### Requirement: Provider Preference Uses Resolution Policy

Provider preference SHALL be represented as Resolution Policy input.

The Runtime SHALL NOT introduce a direct `preferred_provider` API merely as a
renamed replacement for `preferred_backend`.

#### Scenario: Prefer GPU implementation

Given several Providers implement the same Capability

And policy prefers an eligible accelerator Provider

When the Runtime resolves execution

Then Resolution Policy influences candidate ranking

While compatibility and Resource Affinity remain authoritative.

---

### Requirement: No Backend Compatibility Alias

Magnetar SHALL NOT preserve Backend as an alias for Provider in the canonical
public Runtime API.

#### Scenario: Migrate old Backend caller

Given application code uses the historical Backend API

When it migrates to the new Runtime API

Then it adopts Provider and Resolution Policy concepts explicitly

Rather than compiling through a Backend compatibility alias.

---

### Requirement: Native Dynamic Libraries Are Providers

A dynamically loaded native execution extension SHALL be described and managed
as a Provider.

Dynamic loading SHALL NOT imply Plugin or Backend architectural identity.

#### Scenario: Load native provider library

Given the Runtime loads a trusted native dynamic library

And that library satisfies the current Provider factory contract

When registration completes

Then the resulting extension is registered as a Provider.

---

### Requirement: Provider Loading Does Not Define Stable ABI

Removal of Backend and Plugin SHALL NOT imply that the current native Provider
binary interface is stable across compiler versions, Runtime versions, or Rust
toolchains.

Provider ABI stabilization requires a dedicated architectural change.

#### Scenario: Review Provider factory

Given the current Runtime uses a native Provider factory

When this cleanup change is implemented

Then the factory may continue to operate

But its existence is not treated as a long-term stable binary ABI guarantee.

---

### Requirement: Provider and Component Extension Mechanisms Are Distinct

A native trusted extension SHALL use Provider semantics.

A portable sandboxed WebAssembly extension SHALL use Component semantics.

The Runtime SHALL NOT introduce a generic Plugin layer above both mechanisms.

#### Scenario: Classify extensions

Given a CUDA implementation and an OpenTelemetry WASM exporter

When they are classified

Then CUDA is a Provider

And the OpenTelemetry exporter is a Component

And neither requires a generic Plugin abstraction.
