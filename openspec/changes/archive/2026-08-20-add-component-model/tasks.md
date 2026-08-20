# Tasks

## Component API

- [x] Define the `Component` lifecycle trait and component states.
- [x] Define `ComponentMetadata`, including WIT imports, exports, and component dependencies.
- [x] Define `ComponentDescriptor` for discovered component artifacts.

## Lifecycle

- [x] Instantiate a registered component through `ComponentManager`.
- [x] Start components in resolved dependency order.
- [x] Stop started components in reverse start order.
- [x] Destroy stopped components and release manager ownership.

## Discovery

- [x] Discover `.wasm` component artifacts in configured directories.
- [x] Register components while rejecting duplicate names and incompatible contracts.
- [x] Resolve declared dependencies before components start.

## Contracts

- [x] Define a WIT interface value abstraction.
- [x] Validate imported interfaces against host and registered component exports.

## Documentation

- [x] Document component lifecycle and failure semantics.
- [x] Document component architecture and its intentionally deferred runtime integration.
- [x] Document Host and Component responsibilities.
