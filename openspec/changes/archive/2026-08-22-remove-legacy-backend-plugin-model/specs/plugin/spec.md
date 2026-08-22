## REMOVED Requirements

### Requirement: Plugin Discovery

The generic Plugin discovery requirement is removed.

Native extension discovery SHALL use Provider-specific architecture.

Portable WebAssembly artifact discovery SHALL use Component-specific
architecture.

#### Scenario: Historical Plugin discovery

Given the previous specification required generic Plugin discovery

When this change is archived

Then new Runtime behavior no longer depends on a generic Plugin abstraction.

---

### Requirement: Plugin Initialization

The generic Plugin initialization requirement is removed.

Provider initialization SHALL be defined by Provider lifecycle architecture.

Component initialization SHALL be defined by Component lifecycle architecture.

#### Scenario: Initialize extension

Given an extension must be initialized

When the extension is native

Then Provider lifecycle semantics apply

And when the extension is a portable WASM Component

Then Component lifecycle semantics apply.

---

### Requirement: Plugin Version Compatibility

The generic Plugin API compatibility requirement is removed.

Provider compatibility and Component compatibility SHALL be defined
independently according to their respective contracts.

#### Scenario: Validate extension version

Given an extension is loaded

When it is a Provider

Then Provider compatibility rules apply

And when it is a Component

Then WIT and Component compatibility rules apply.

---

### Requirement: Plugin Metadata

The generic Plugin metadata requirement is removed.

Providers and Components SHALL expose metadata appropriate to their respective
architectural models.

#### Scenario: Query extension metadata

Given Runtime metadata is required for an extension

When the extension is native

Then Provider metadata is used

And when the extension is portable WASM

Then Component metadata is used.

---

### Requirement: General Plugin Interface

The generic `Plugin` interface is removed.

Magnetar SHALL NOT use a generic Plugin interface to register Backend,
Provider, Component, model, kernel, compiler, scheduler, or telemetry
contributions.

#### Scenario: Register native extension

Given a native accelerator implementation

When it is registered

Then it implements Provider semantics directly

And does not first implement a generic Plugin interface.

---

### Requirement: Extensible Plugin Registry

The generic Plugin Registry is removed.

Magnetar SHALL NOT use a single generic contribution Registry for unrelated
native and portable extension categories.

Provider registration SHALL use Provider architecture.

Component registration SHALL use Component architecture.

Future model, tool, compiler, scheduler, or other extension mechanisms SHALL be
defined through their appropriate dedicated architecture.

#### Scenario: Register observability exporter

Given a portable observability exporter is installed

When Magnetar handles the exporter

Then it is managed as an Observability Component

And is not registered through a generic Plugin Registry.

---

### Requirement: Plugin Lifecycle

The generic Plugin lifecycle requirement is removed.

Provider lifecycle and Component lifecycle SHALL remain independent.

#### Scenario: Runtime shutdown

Given Providers and Components are active

When the Runtime shuts down

Then each is shut down according to its own lifecycle contract

And no generic Plugin lifecycle is required.
