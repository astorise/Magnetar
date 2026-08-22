# Remove Legacy Backend and Plugin Model

## Why

Magnetar's initial Runtime bootstrap introduced `Backend` as the abstraction for
hardware execution.

A later architecture refactor introduced `Provider` as the canonical native
extension mechanism.

The current Runtime therefore still contains two overlapping native execution
models:

```text
Runtime
├── Backend
└── Provider
```

This dual model no longer matches the canonical architecture.

The canonical Magnetar execution model is:

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

Providers implement Capabilities.

Providers expose Devices.

The Runtime resolves Providers and Devices according to Capability
compatibility, Resource Affinity, Provider advertisements, health, and
Resolution Policy.

A separate Backend abstraction creates an alternative path around this model
and makes ownership ambiguous.

The Runtime also still contains historical APIs and state related to Backend
selection, including concepts such as:

- `Backend`
- backend registration
- preferred backend configuration
- backend selection
- backend names associated with execution contexts

These concepts conflict with the current Provider-based architecture.

The canonical OpenSpec tree also still contains the historical `plugin`
specification.

That specification describes:

- plugin discovery
- plugin initialization
- plugin compatibility
- plugin metadata
- a general Plugin interface
- a plugin Registry
- backend contributions
- plugin lifecycle

The Plugin abstraction is now redundant.

Native extensions are Providers.

Portable WebAssembly extensions are Components.

Maintaining a third generic extension mechanism would create overlapping
responsibilities between:

```text
Plugin
Provider
Component
```

and would make future work on the WASM Component Runtime, model Components,
observability Components, Provider lifecycle, and Tachyon distribution more
difficult.

Magnetar is still pre-1.0.

This is therefore the appropriate point to remove the legacy abstractions
rather than preserving compatibility aliases that would perpetuate them.

## What Changes

This change removes Backend and Plugin as active Runtime architecture.

### Backend Removal

The Runtime SHALL remove the legacy `Backend` abstraction.

The implementation SHALL remove Backend-specific:

- traits
- registries
- registration APIs
- lookup APIs
- builder state
- configuration
- execution-context fields
- selection logic
- tests
- documentation

The Runtime SHALL NOT introduce:

```text
type Backend = Provider
```

or another long-lived compatibility alias.

Native hardware execution SHALL occur exclusively through Providers and their
Devices.

### Provider-Only Native Execution

The canonical native execution path SHALL become:

```text
Runtime
    |
    v
ProviderRegistry
    |
    v
Provider
    |
    v
Device
```

There SHALL NOT be a parallel Backend registry.

A Runtime MAY initialize without any Provider.

Provider availability SHALL only become relevant when execution requires a
Capability that must be resolved.

### Configuration

Backend-specific configuration SHALL be removed.

A configuration field such as:

```text
preferred_backend
```

SHALL NOT simply be renamed to:

```text
preferred_provider
```

as a direct execution selector.

Provider preferences belong to Resolution Policy.

Configuration MAY select or configure a Resolution Policy, but portable
callers SHALL NOT bypass resolution by naming a Provider or Device.

### Execution Context

Execution contexts SHALL NOT carry legacy Backend identity.

Where execution identity must be recorded, the Runtime SHALL use existing
Provider, Device, Capability, plan, or affinity bindings.

Portable execution semantics SHALL remain independent from a Backend name.

### Plugin Removal

The active canonical Plugin specification SHALL be removed.

The following historical Plugin requirements SHALL cease to be canonical:

- Plugin Discovery
- Plugin Initialization
- Plugin Version Compatibility
- Plugin Metadata
- General Plugin Interface
- Extensible Plugin Registry
- Plugin Lifecycle

Historical archived OpenSpec changes SHALL NOT be rewritten.

The archived `add-plugin-system` change remains historical evidence of project
evolution.

### Plugin Responsibility Migration

Former Plugin responsibilities SHALL be classified according to their actual
architectural role.

Native hardware or execution extensions SHALL be Providers.

Portable WASM extensions SHALL be Components.

Examples:

```text
Historical concept                 Canonical concept

CUDA backend plugin       ->       CUDA Provider
CPU backend plugin        ->       CPU Provider
kernel provider plugin    ->       Provider responsibility
telemetry plugin          ->       Observability Component
model architecture plugin ->       Model Component or model Runtime module
tool plugin               ->       Tool Component
compiler extension        ->       Runtime/compiler architecture
```

This change SHALL NOT create new stable contracts for future compiler, model,
agent, or tool extensions.

Those domains remain subject to dedicated OpenSpec changes.

### Provider Loading

Existing native Provider loading MAY continue to use dynamic libraries.

Dynamic loading SHALL be described as Provider discovery/loading rather than
Plugin or Backend loading.

Provider loading remains a trusted native boundary.

This change SHALL NOT stabilize the Provider binary ABI.

### Resolution

All Provider selection SHALL continue through the existing Capability and
Resolution Policy model.

Backend removal SHALL NOT create direct Provider selection APIs for Components.

Resource Affinity SHALL remain authoritative.

### Documentation and Tests

Current documentation SHALL stop presenting Backend and Plugin as active
architectural concepts.

Tests SHALL verify that:

- Runtime initialization works without Providers
- Providers can be registered independently
- Devices are exposed through Providers
- Capability resolution selects Providers
- Resource Affinity remains enforced
- no Backend registry remains
- no Plugin registry remains
- no direct backend-selection configuration remains

## Non-Goals

This change does not:

- redesign the Provider lifecycle
- stabilize a native Provider ABI
- implement the real WASM Component Runtime
- tighten the Compute WIT data-movement boundary
- change Resource Affinity semantics
- redesign Resolution Policy
- introduce model execution
- introduce inference scheduling
- introduce Component artifact distribution
- change Provider health semantics

Those concerns are handled by dedicated follow-up changes.

## Impact

This is a breaking Rust API cleanup.

Code using the legacy Backend API must migrate to Providers.

Code using the historical Plugin abstraction must migrate to either Provider or
Component semantics depending on its role.

The Runtime becomes conceptually simpler:

```text
Before

Runtime
├── Backend Registry
├── Provider Registry
├── Plugin concepts
└── Component concepts


After

Runtime
├── Provider Registry
│    └── Provider
│         └── Device
│
└── Component system
     └── portable WASM Components
```

This establishes the architecture required before Provider lifecycle,
Component Runtime, model execution, and AI inference work continue.