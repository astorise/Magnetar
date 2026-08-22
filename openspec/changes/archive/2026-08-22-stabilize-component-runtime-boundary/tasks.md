# Tasks

## 1. OpenSpec Hygiene

- [x] Validate and normalize the OpenSpec delta files so the change parses under `openspec validate --strict`.
- [x] Keep implementation tasks grouped by deliverable rather than by every individual requirement sentence.

## 2. Component Runtime Domain Model

- [x] Inventory the existing `component` module and tests before editing.
- [x] Refactor component metadata away from direct Component-name dependencies and toward WIT import/export contracts.
- [x] Introduce engine-neutral runtime concepts for definitions, prepared Components, instances, link plans, resource limits, traps, interruption, and stable Component Runtime errors.
- [x] Keep concrete engine state opaque and avoid public Wasmtime-specific types.

## 3. Component Engine Boundary

- [x] Introduce an engine-neutral `ComponentEngine` abstraction.
- [x] Add a mock/test engine that can prepare, instantiate, trap, interrupt, fail limits, and destroy instances without Wasmtime.
- [x] Ensure ComponentEngine remains separate from Provider and Magnetar Capability registration.

## 4. Runtime Linking and Lifecycle

- [x] Refactor `ComponentManager` around Runtime-owned validation, authorization, link-plan construction, preparation, instantiation, invocation coordination, and destruction.
- [x] Remove canonical direct Component-name dependency resolution and dependency-cycle handling.
- [x] Remove mandatory generic `start`/`stop` lifecycle assumptions.
- [x] Preserve Component discovery behavior until the artifact model supersedes it.

## 5. Tests

- [x] Cover missing import rejection, unauthorized import rejection, valid Capability linking, and fail-closed ambient authority.
- [x] Cover definition/instance separation, multiple instances per prepared definition, instance isolation, destruction, and invocation after destruction.
- [x] Cover trap normalization, interruption normalization, and required resource-limit enforcement failure with the mock engine.
- [x] Verify Capability linking does not select a concrete Provider and does not expose Provider handles.

## 6. Documentation and Validation

- [x] Update Component architecture documentation for Runtime versus Engine, definitions versus instances, link plans, no ambient authority, and no named Component dependencies.
- [x] Run formatting.
- [x] Run compilation checks.
- [x] Run Clippy.
- [x] Run complete tests.
- [x] Run OpenSpec validation.
