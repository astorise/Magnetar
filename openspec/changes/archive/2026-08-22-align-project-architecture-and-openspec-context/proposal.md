# Align Project Architecture and OpenSpec Context

## Why

Magnetar's architecture has matured significantly since the initial Runtime
bootstrap.

The project is no longer accurately described as a hardware abstraction layer
centered around Backends.

The canonical architecture is now based on:

- Runtime
- Component
- Capability
- Provider
- Device
- Resource Affinity
- Resolution Policy

The project scope has also become clearer.

Magnetar is intended to become a standalone universal AI Runtime capable of
owning local AI execution, including future responsibilities such as:

- model loading
- model residency
- tokenization
- prompt formatting
- generation
- streaming
- continuous batching
- KV cache management
- adapters
- quantization
- multi-device execution
- agent execution
- tool execution
- observability
- local CLI usage
- service API usage

Magnetar is expected to replace the AI inference implementation currently
located in Tachyon.

Tachyon and Magnetar SHALL remain independent architectural layers.

Magnetar owns AI execution.

Tachyon owns distributed service orchestration.

Tachyon MAY distribute Magnetar-compatible WASM Components and artifacts, but
Magnetar SHALL NOT depend on Tachyon for standalone operation.

The repository documentation and OpenSpec project context do not yet fully
encode these architectural decisions.

Some documentation still uses historical concepts such as Backend as a primary
Runtime abstraction.

The OpenSpec project configuration also lacks sufficient domain context and
architectural rules to prevent future changes from accidentally reintroducing
obsolete concepts or violating established boundaries.

Because Magnetar development is specification-driven, architectural invariants
must be encoded into repository-owned OpenSpec context rather than relying on
conversation history or contributor knowledge.

## What Changes

This change establishes a canonical project architecture description and makes
it authoritative for future OpenSpec work.

The change SHALL:

- define the canonical Magnetar architecture
- define canonical terminology
- define deprecated terminology
- document the Runtime / Component / Capability / Provider / Device model
- document Resource Affinity and Resolution Policy roles
- define the Component versus Provider boundary
- define the Magnetar versus Tachyon responsibility boundary
- define the future Magnetar AI Runtime scope
- define the relationship between Magnetar and `magnetar-cli`
- define external Component distribution without introducing a Tachyon
  dependency
- distinguish Component artifacts from Model artifacts
- document coarse-grained WIT execution boundaries
- document security and authority principles for Components
- document native trust assumptions for Providers
- update the project README to reflect the canonical architecture
- introduce a canonical architecture overview document
- populate `openspec/config.yaml` with project-specific context
- add OpenSpec authoring rules that preserve architectural invariants
- document which architectural concepts are stable and which remain future work

The canonical execution model SHALL be:

```text
Component
    |
    | imports Capability
    v
Runtime
    |
    | Resolution Policy
    v
Provider
    |
    v
Device
```

Components SHALL request portable Capabilities.

Components SHALL NOT directly select native Providers or Devices.

Providers SHALL implement Capabilities and expose Devices.

The Runtime SHALL perform resolution.

Resource Affinity SHALL constrain resolution when live resources or state are
bound to a Provider, Device, artifact, execution context, model, adapter, cache,
or other resource identity.

The project SHALL stop describing Backend, Plugin, or Host as primary
architectural concepts.

This change defines documentation and specification context only.

Removal of the remaining legacy Backend and Plugin implementation is handled by
a dedicated follow-up change.

## Impact

Future OpenSpec changes will receive sufficient repository-owned context to
preserve Magnetar architecture without relying on external conversation state.

README documentation will match the actual architectural direction.

The project will have a stable vocabulary for Runtime, Components,
Capabilities, Providers, Devices, artifacts, model execution, and distributed
integration.

Future AI inference work can be introduced without ambiguity over whether a
responsibility belongs to Magnetar or Tachyon.

Future Component work can also proceed without confusing portable WASM
Components with trusted native Providers.

No Runtime execution semantics change as part of this proposal.