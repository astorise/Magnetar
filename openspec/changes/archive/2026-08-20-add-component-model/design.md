# WebAssembly Component Model Design

## Scope

This change defines host-side contracts and lifecycle orchestration for portable
WebAssembly Components. It deliberately does not execute WebAssembly, parse WIT,
or define component implementations. A future runtime integration will adapt
these contracts to a concrete WebAssembly engine.

## Model

`ComponentMetadata` identifies a component and declares the WIT interfaces it
imports and exports plus its component dependencies. `ComponentDescriptor`
combines that metadata with the component artifact path discovered on disk.

`Component` is the host-facing lifecycle contract. A component is instantiated,
started, stopped, and destroyed by `ComponentManager`. Components are started in
dependency order and stopped/destroyed in reverse start order. Failed startup
leaves already-started components stopped.

## Contracts and Compatibility

`WitInterface` represents an interface by its package-qualified name and
version. A component can be registered only when every imported interface is
available from a host or another registered component. Exported interfaces make
those contracts available to dependent components. Exact names and versions are
required; WIT parsing and semver range negotiation remain future work.

## Discovery

Discovery scans configured directories for `.wasm` files and produces stable,
sorted paths. Descriptor construction and runtime loading remain explicit so a
future manifest format can be added without changing discovery.

## Errors and Responsibilities

The manager rejects duplicate component names, missing dependencies,
unsatisfied interfaces, and invalid lifecycle transitions through
`ComponentError`. Hosts provide WIT interfaces and hardware resources;
components consume those interfaces and remain free of hardware-specific APIs.
