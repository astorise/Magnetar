# Stabilize Component Runtime Boundary

## Why

Magnetar defines portable WebAssembly Components as one of its canonical
architectural concepts.

Components consume portable WIT contracts while trusted native execution remains
implemented through Providers.

The current Component implementation was intentionally introduced as an
engine-neutral prototype.

It currently models concepts such as:

- Component metadata
- WIT imports and exports
- Component registration
- Component discovery
- Component dependencies
- instantiate/start/stop/destroy lifecycle
- ComponentManager

However, Magnetar does not yet have a real WebAssembly Component Model engine.

The next implementation phase will introduce one.

Before selecting and integrating a concrete engine such as Wasmtime, Magnetar
needs a stable internal boundary separating:

```text
Magnetar Runtime semantics
           |
           v
Component Runtime abstraction
           |
           v
Component Engine adapter
           |
           v
Concrete WASM engine
```

Without this boundary, Wasmtime-specific concepts such as:

- `Engine`
- `Store`
- `Linker`
- compiled Component handles
- traps
- epoch interruption
- fuel
- resource tables
- generated bindings

could leak throughout Magnetar.

That would make Magnetar's architecture dependent on one implementation and
would complicate testing, future engine replacement, sandbox policy, and
Component distribution.

The current Component model also contains two assumptions that should be
corrected before a real engine is introduced.

First, Components currently declare dependencies on other Components by name.

The canonical Magnetar architecture instead requires Components to express
dependencies through WIT imports and Capabilities.

A Component should depend on:

```text
magnetar:compute/run
magnetar:observability/stream
magnetar:tool/filesystem
```

rather than:

```text
cuda-component
otel-component
filesystem-component
```

Second, the prototype assumes that every Component has generic
`start()` and `stop()` lifecycle operations.

The WebAssembly Component Model does not define universal Magnetar-specific
start and stop exports.

Generic Component lifecycle SHALL therefore describe Runtime instance ownership
rather than invent mandatory application-level lifecycle functions.

A Component requiring application-specific initialization or shutdown MAY
define those semantics through an explicit WIT contract.

## What Changes

This change defines the stable internal Component Runtime boundary.

It introduces the conceptual responsibilities of:

- Component Runtime
- Component Engine
- Component Definition
- Prepared Component
- Component Instance
- Component Store
- Component Link Plan
- Component Import Requirement
- Component Export Description
- Component Resource Limits
- Component invocation context
- Component interruption/cancellation
- Component Trap
- stable Component Runtime errors

These names describe architectural concepts.

The exact Rust type decomposition MAY differ where implementation ergonomics
justify it.

### Component Runtime and Component Engine

Magnetar SHALL distinguish:

```text
Component Runtime
```

from:

```text
Component Engine
```

The Component Runtime belongs to Magnetar.

It owns:

- Component registration
- contract inspection
- compatibility validation
- import requirements
- authorization
- Capability linking
- instance lifecycle
- Runtime resource ownership
- invocation coordination
- error normalization
- observability integration

The Component Engine is an implementation adapter.

It owns engine-specific mechanics such as:

- WebAssembly validation
- compilation
- engine-native prepared representation
- Store creation
- linker construction
- instantiation
- export binding
- execution
- interruption
- trap extraction
- engine-native resource cleanup

The engine SHALL NOT own Magnetar Capability resolution policy.

### Engine Independence

Public Magnetar Component domain types SHALL NOT expose concrete Wasmtime types.

The following SHALL remain behind the Component Engine boundary:

```text
wasmtime::Engine
wasmtime::component::Component
wasmtime::component::Linker
wasmtime::Store
wasmtime::component::Resource
wasmtime::Trap
```

or equivalent objects from another engine.

A future Wasmtime implementation SHALL adapt these concepts to Magnetar's
engine-neutral contracts.

### Component Definition

Magnetar SHALL distinguish a Component definition from a running Component
instance.

A Component definition represents validated information describing executable
Component code and its contracts.

It MAY include:

- logical identity
- metadata
- WIT imports
- WIT exports
- compatibility information

Artifact digests, signatures, publishers, and supply-chain trust are deferred to
the dedicated Component Artifact and Trust Model change.

### Prepared Component

A Component Engine MAY transform validated Component bytes into an optimized or
compiled representation.

That representation SHALL be opaque outside the Component Engine adapter.

Conceptually:

```text
Component bytes
      |
      v
contract validation
      |
      v
ComponentEngine::prepare
      |
      v
PreparedComponent
```

`PreparedComponent` SHALL NOT be exposed through WIT.

It SHALL NOT be treated as a portable artifact.

### Component Instance

Instantiation SHALL create an isolated Component Instance.

A Component Instance SHALL have a Runtime-owned identity independent from the
Component definition.

Multiple instances MAY originate from the same Component definition.

Conceptually:

```text
ComponentDefinition
        |
        +----> ComponentInstance A
        |
        +----> ComponentInstance B
        |
        +----> ComponentInstance C
```

Instance-local mutable state SHALL NOT be assumed to be shared between these
instances.

### Component Store

Each Component Instance SHALL execute within engine-managed execution state
conceptually represented as a Component Store.

The Store MAY contain:

- engine execution state
- WIT resource tables
- Component-local state
- host-call state
- cancellation state
- invocation context
- limits and quotas
- authorized Capability endpoints

The concrete Store representation SHALL remain engine-private.

Unrelated Components SHALL NOT implicitly share mutable Store state.

### Linking

Component linking SHALL be driven by WIT imports.

A Runtime-created Component Link Plan SHALL map each authorized Component import
to an appropriate Runtime endpoint.

For a Magnetar Capability, the path is conceptually:

```text
Component
    |
    | WIT import
    v
Component Link Plan
    |
    v
Runtime Capability Endpoint
    |
    +--> Runtime service
    |
    `--> Capability Resolution
             |
             v
          Provider
             |
             v
           Device
```

Linking a Capability SHALL NOT permanently select a Provider.

For example, linking:

```text
magnetar:compute/run
```

means that the Component can invoke the Runtime Compute Capability.

It does not mean:

```text
this Component is permanently linked to CUDAProvider
```

Provider and Device resolution continues to occur according to Runtime policy,
Resource Affinity, and execution state.

### Remove Named Component Dependencies

The canonical Component model SHALL no longer require direct dependencies on
Component names.

The current pattern:

```text
Component A
    dependencies:
        Component B
```

SHALL be replaced by:

```text
Component A
    imports:
        interface X
```

The Runtime determines how authorized imports are satisfied.

Component identity SHALL NOT become a service locator.

### Component Exports

A Component MAY export WIT interfaces.

An exported interface SHALL NOT automatically become a globally available
Magnetar Capability merely because it exists.

The Runtime SHALL explicitly decide whether and how an export is exposed.

Automatic Component-to-Component linking based solely on matching exports is
not part of the canonical model.

Explicit Component composition MAY be introduced later.

### No Ambient Authority

Instantiation SHALL use an explicit Link Plan.

A Component SHALL receive only interfaces that the Runtime intentionally links.

The Runtime SHALL NOT automatically grant:

- filesystem access
- network access
- environment variables
- process execution
- clocks beyond explicitly exposed semantics
- secrets
- sockets
- host commands
- arbitrary WASI interfaces

merely because the concrete WASM engine supports them.

The detailed value-level authority and scoping model belongs to the later
`define-component-authority-scoping-model` change.

This change establishes the foundational fail-closed rule:

```text
not linked
    =
not available
```

### WASI

WASI SHALL NOT be ambient.

If a Component requires a WASI interface, that interface SHALL be explicitly
allowed and linked according to Runtime policy.

The Component Engine SHALL NOT automatically install a broad default WASI
environment.

### Component Lifecycle

Magnetar SHALL separate definition lifecycle from instance lifecycle.

A Component definition MAY conceptually move through:

```text
registered
    |
    v
validated
    |
    v
prepared
    |
    +----> failed
    |
    v
removed
```

A Component Instance MAY conceptually move through:

```text
instantiating
      |
      v
    ready
      |
      +------> failed
      |
      v
   draining
      |
      v
  destroyed
```

The exact Rust enum names MAY differ.

A generic Component SHALL NOT be required to export universal `start()` or
`stop()` functions.

Once successfully instantiated and linked, an instance becomes available for
invocation according to its exported interfaces.

Runtime shutdown SHALL prevent new invocations and destroy Component instances
according to Runtime lifecycle policy.

### Invocation

Component invocation SHALL occur through WIT contract-specific adapters.

Magnetar SHALL NOT require a universal dynamically typed API such as:

```text
invoke(name, Vec<DynamicValue>)
```

as its canonical public Component API.

Generated bindings or typed Runtime adapters MAY be used.

This preserves strong contract semantics.

### Asynchronous Capability Calls

The Component Engine boundary SHALL support host Capability implementations that
may complete asynchronously.

The abstraction SHALL NOT require every host function to block a native thread
until Provider work completes.

The exact Rust async runtime is not standardized by this change.

### Cancellation and Interruption

The Runtime SHALL be able to request interruption of Component execution.

Interruption MAY be used for:

- caller cancellation
- Runtime shutdown
- execution deadline
- resource-policy violation
- administrative termination

Component interruption SHALL map to stable Magnetar Component Runtime errors.

Engine-specific interruption mechanisms such as fuel exhaustion or epoch
interruption SHALL remain implementation details.

Cancelling execution of the WASM Component SHALL NOT automatically imply that
already-submitted Provider operations can be cancelled.

Provider operation cancellation SHALL continue to follow Provider execution
semantics.

### Traps

Engine-specific traps SHALL be normalized into stable Magnetar Component errors.

A trap MAY include a redacted diagnostic.

Portable or public Runtime APIs SHALL NOT depend on a concrete engine's Trap
type or message format.

### Resource Ownership

WIT resources created for a Component Instance SHALL have explicit ownership
and lifetime.

The Runtime SHALL prevent one Component Instance from forging access to another
instance's resource.

Dropping a Component Instance SHALL release instance-owned engine resources.

Runtime resources with independent ownership SHALL follow their own lifecycle
and SHALL NOT be invalidated merely because an unrelated engine handle was
dropped.

### Resource Limits

The Component Runtime SHALL support semantic resource limits.

Limits MAY include:

- maximum Component memory
- execution deadline
- maximum concurrent invocations
- maximum instance count
- engine-specific execution budget

Portable policy SHALL describe the required limit semantics.

Concrete enforcement mechanisms remain engine-specific.

For example, a Wasmtime implementation MAY use:

- Store limits
- epoch interruption
- fuel
- pooling configuration

without exposing those mechanisms as canonical Magnetar concepts.

### Concurrency

The Runtime SHALL respect the concurrency guarantees of the selected Component
Engine and Component Instance.

The Runtime SHALL NOT assume that engine Store state can be concurrently mutated
from multiple threads.

A Component Instance SHALL NOT receive concurrent re-entry unless the engine
adapter and Runtime invocation model explicitly support it.

### Engine Capabilities

A Component Engine MAY advertise engine-level implementation capabilities such
as:

- Component Model support
- asynchronous host-call support
- interruption support
- resource-limit enforcement support

These engine capabilities are not Magnetar Capabilities.

`ComponentEngine` SHALL NOT become a Provider.

### Failure Isolation

A Component trap or Component-level failure SHALL fail the affected invocation
or instance according to error severity.

It SHALL NOT directly mutate Provider resolution, Resource Affinity, or
Scheduler state except through normal Runtime error propagation.

A Component failure SHALL NOT imply Provider failure.

Likewise, a Provider failure SHALL NOT imply that the WASM engine itself has
failed.

### Observability

Component Runtime operations SHOULD integrate with Magnetar Runtime
observability.

Observations MAY include:

- Component definition identity
- Component instance identity
- preparation duration
- instantiation duration
- invocation duration
- trap category
- cancellation
- resource-limit violations

Observability SHALL NOT expose engine-native pointers, Store addresses, raw
resource handles, or secret data.

### Concrete Engine

This change SHALL NOT select the canonical production engine implementation.

The following change:

```text
implement-wasm-component-runtime
```

will provide the first concrete implementation, expected to use the WebAssembly
Component Model through Wasmtime unless implementation evidence requires another
choice.

The architecture defined here SHALL remain valid even if the concrete engine is
replaced.

## Non-Goals

This change does not:

- integrate Wasmtime
- define Component artifact digests
- define Component signatures
- define publishers or trust roots
- define Tachyon Component distribution
- define value-level filesystem scopes
- define network allowlists
- define secret scopes
- define tool permissions
- define model Components
- define agent Components
- define Component hot reload
- define cross-node Component execution
- make ComponentEngine a Provider
- expose engine-native handles through WIT
- stabilize a generic dynamic invocation ABI

## Impact

The existing prototype ComponentManager will be refactored around an
engine-independent Component Runtime boundary.

Direct Component-name dependencies will disappear from the canonical model.

Generic `start()` and `stop()` Component lifecycle assumptions will disappear.

WIT imports become the authoritative dependency description.

The resulting architecture is:

```text
                    Magnetar Runtime
                          |
                          v
                  Component Runtime
                    /          \
                   /            \
          contracts/policy    instances
                 |                |
                 v                v
           ComponentLinkPlan  ComponentEngine
                 |                |
                 v                v
       Runtime Capability      concrete WASM
           endpoints              engine
                 |
          +------+------+
          |             |
          v             v
     Runtime service   Provider resolution
                            |
                            v
                          Device
```

This creates the stable boundary required before a real WebAssembly Component
Model engine is introduced.