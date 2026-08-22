# Implement WASM Component Runtime

## Why

Magnetar defines portable WebAssembly Components as a canonical extension
mechanism.

Previous changes established that:

- Components are portable WASM Components
- Components consume WIT Capabilities
- Components are distinct from native Providers
- Components do not directly select Providers or Devices
- Components do not receive native handles
- Components receive no ambient authority
- Component linking is Runtime-owned
- Component execution is mediated through an engine-neutral `ComponentEngine`
  boundary

The current implementation still lacks a real WebAssembly Component Model
engine.

To support future Model Components, Observability Components, Tool Components,
Agent Components, and Tachyon-distributed Magnetar Components, Magnetar now
needs its first concrete Component Runtime implementation.

This change introduces the first production Component Engine adapter.

The expected implementation is based on Wasmtime's WebAssembly Component Model.

Wasmtime is selected as the initial concrete engine because it provides:

- WebAssembly Component Model support
- WIT-based linking
- typed host bindings
- async host-function support
- resource-table support
- configurable execution limits
- interruption mechanisms
- mature Rust integration

However, Wasmtime SHALL remain an implementation detail behind Magnetar's
`ComponentEngine` boundary.

The canonical Magnetar architecture remains:

```text
Component
    |
    | imports Capability
    v
Runtime
    |
    | Runtime-owned Link Plan
    v
ComponentEngine adapter
    |
    v
Concrete WASM engine
```

Provider and Device resolution remain Runtime responsibilities.

Component instantiation does not pin a Provider.

Capability invocation may later resolve Providers according to Runtime policy,
Resource Affinity, Provider advertisements, health, and execution state.

## What Changes

This change implements the first concrete WebAssembly Component Runtime.

The change SHALL:

- add a concrete Wasmtime-based `ComponentEngine` adapter
- validate WASM Component artifacts before instantiation
- prepare executable Components through the engine adapter
- instantiate Components using Runtime-owned Link Plans
- link only explicitly authorized imports
- support Components importing Magnetar Capabilities
- support typed or generated WIT host adapters
- support async host calls where required
- support instance identity
- support instance-local Store state
- support interruption/cancellation where configured
- normalize traps into stable Magnetar errors
- enforce or reject configured resource limits
- provide test fixtures using real WASM Components
- keep Wasmtime-specific types out of Magnetar public APIs
- preserve no-ambient-authority semantics

### Concrete Engine

The first concrete engine SHOULD be implemented using Wasmtime.

The implementation MAY be named conceptually:

```text
WasmtimeComponentEngine
```

or equivalent.

The engine SHALL implement the previously defined `ComponentEngine` boundary.

It SHALL NOT replace the abstract Component Runtime with direct Wasmtime usage.

### Dependencies

The `magnetar-runtime` crate MAY add dependencies required for Wasmtime
Component Model execution.

These MAY include:

- `wasmtime`
- `wasmtime-wasi`, if explicitly required for controlled WASI linking
- `wit-bindgen` or generated binding support where appropriate
- test helper crates required to build Component fixtures

Dependency versions SHALL be pinned according to repository policy.

Optional feature flags MAY be introduced so that the concrete WASM engine can
be enabled or disabled.

A feature name such as:

```text
wasmtime-component-engine
```

MAY be used.

The default feature policy SHALL be explicit.

### Engine Boundary

Wasmtime-native types SHALL remain private to the adapter module.

Public or canonical Magnetar APIs SHALL NOT expose:

```text
wasmtime::Engine
wasmtime::Config
wasmtime::Store
wasmtime::component::Component
wasmtime::component::Linker
wasmtime::component::Instance
wasmtime::Trap
wasmtime::Error
```

or equivalent engine-owned objects.

Magnetar public types SHALL continue to use Magnetar-owned abstractions such as:

- ComponentDefinition
- PreparedComponent
- ComponentInstance
- ComponentInstanceId
- ComponentLinkPlan
- ComponentRuntimeError
- ComponentTrap
- ComponentResourceLimits

The exact Rust type names may differ, but the ownership boundary is normative.

### Component Preparation

The engine SHALL prepare validated Component bytes before instantiation.

Preparation MAY include:

- parsing
- WebAssembly validation
- Component Model validation
- engine compilation
- pre-instantiation metadata extraction
- engine-specific optimization

Preparation failures SHALL map to stable Magnetar Component errors.

A prepared Component SHALL be opaque outside the engine adapter.

### WIT Contract Validation

The Runtime SHALL validate Component WIT imports and exports before
instantiation.

Validation SHALL ensure that:

- required imports are known
- required imports are compatible
- required imports are authorized
- unsupported mandatory imports fail closed
- exported interfaces can be inspected
- unsupported Component shape is rejected before invocation

This validation SHALL be separate from engine compilation errors.

### Link Plan Execution

The Runtime SHALL construct a Component Link Plan.

The Wasmtime adapter SHALL translate the approved Link Plan into engine-native
linker state.

Only imports present in the approved Link Plan SHALL be linked.

No broad default WASI environment SHALL be installed automatically.

### Magnetar Capability Host Adapters

The implementation SHALL provide at least one real Magnetar host adapter
suitable for an end-to-end Component fixture.

The initial fixture MAY use a small test Capability rather than the complete
Compute Capability if Compute v2 bindings are not yet fully generated.

However, the design SHALL demonstrate the same path used for real Magnetar
Capabilities:

```text
Component import
      |
      v
Runtime Link Plan
      |
      v
Host adapter
      |
      v
Runtime Capability endpoint
```

When the imported Capability is Provider-backed, the host adapter SHALL call the
Runtime service rather than exposing a Provider directly.

### No Provider Pinning at Link Time

Linking a Capability import SHALL NOT select a concrete Provider or Device.

For example, linking:

```text
magnetar:compute/run
```

SHALL link a Runtime Compute endpoint.

Provider selection still occurs during Runtime execution according to:

- Capability compatibility
- Resource Affinity
- Provider advertisements
- Resolution Policy
- health/readiness/pressure where applicable
- execution planning

### Async Host Calls

The Wasmtime adapter SHALL support asynchronous host calls where required by
Magnetar Capabilities.

The implementation SHALL NOT require long-running Provider work to block a
host thread across the entire native execution.

Where the concrete engine requires async configuration, it SHALL be enabled
inside the adapter.

The public Magnetar Component Runtime boundary SHALL remain independent from a
specific async runtime.

### Store and Instance State

Each Component Instance SHALL have isolated engine Store state.

The Runtime SHALL associate the engine Store with a Runtime-owned
ComponentInstanceId.

Two instances created from the same Component definition SHALL not implicitly
share mutable Store state.

The adapter SHALL respect engine rules for Store mutability and concurrency.

### Resource Tables

The engine may maintain WIT resource tables.

Resource table entries SHALL remain engine-private.

They SHALL NOT become stable Magnetar resource identifiers.

When Component-owned WIT resources map to Runtime resources, the Runtime SHALL
validate ownership and lifetime according to Magnetar resource rules.

### Interruption and Cancellation

The adapter SHALL support Runtime-requested interruption where the concrete
engine supports it.

Interruption MAY be used for:

- explicit cancellation
- deadline expiration
- Runtime shutdown
- administrative termination
- resource-policy violation

Wasmtime-specific mechanisms such as fuel or epoch interruption SHALL remain
implementation details.

Cancellation of Component execution SHALL remain distinct from cancellation of
Provider work already submitted by a host call.

### Traps and Errors

Wasmtime traps and errors SHALL be normalized into stable Magnetar Component
Runtime errors.

Errors SHALL distinguish at least:

- validation failure
- preparation failure
- link failure
- instantiation failure
- missing import
- unauthorized import
- invocation trap
- invocation interruption
- resource-limit violation
- engine failure

Diagnostics MAY include redacted engine messages.

Diagnostics SHALL NOT expose native pointers, Store addresses, raw engine
handles, secrets, or private host state.

### Resource Limits

The Wasmtime adapter SHALL support configured Component resource limits where
feasible.

Limits MAY include:

- memory limits
- execution deadlines
- maximum instances
- maximum concurrent invocations
- fuel or epoch budget as implementation detail

If Runtime policy requires a limit and the selected engine configuration cannot
enforce it, instantiation or execution SHALL fail closed.

The Runtime SHALL NOT silently ignore required safety policy.

### WASI

WASI SHALL be linked only when explicitly authorized.

The adapter SHALL NOT provide ambient filesystem, network, environment,
standard IO, clocks, randomness, or process-related interfaces merely because
Wasmtime or WASI supports them.

Detailed value-level authority scoping is deferred to
`define-component-authority-scoping-model`.

This change SHALL still preserve the fail-closed rule:

```text
not authorized
    =
not linked
```

### Test Components

This change SHALL introduce real WASM Component fixtures.

Fixtures SHOULD cover:

- valid Component with authorized import
- Component with missing import
- Component with unauthorized import
- Component with export invocation
- Component that traps
- Component interrupted by deadline or cancellation where feasible
- Component denied ambient WASI access

Fixtures MAY be small Rust, WAT, or generated Component artifacts depending on
repository tooling.

The test fixture build process SHALL be reproducible in CI.

### Observability

Component engine operations SHOULD emit Runtime observations for:

- validation
- preparation
- instantiation
- linking
- invocation
- traps
- interruption
- limit violations
- destruction

Observability SHALL not control execution.

Slow or failing observability exporters SHALL not alter Component correctness.

### Feature Gating

If the Wasmtime dependency materially increases build time or platform
requirements, the implementation MAY place the concrete engine behind a Cargo
feature.

Engine-neutral Component Runtime tests SHOULD remain available without the
Wasmtime feature where possible.

Wasmtime end-to-end tests SHALL run in CI when the feature is enabled.

### Platform Support

The engine integration SHALL compile on supported CI platforms unless a
platform-specific limitation is explicitly documented.

At minimum, Linux CI SHALL execute end-to-end Component tests.

Windows and macOS SHALL compile the integration and run tests where supported.

### Future Distribution

This change does not define Component artifact trust, signatures, digest
addressing, external registries, or Tachyon distribution.

It only establishes executable local Component support.

Future changes will define:

- Component artifact and trust model
- authority scoping
- Component distribution contract
- Tachyon-Magnetar integration

## Non-Goals

This change does not:

- define Component signatures
- define Component digest addressing
- define Component publisher identity
- define Tachyon Component distribution
- define filesystem scope policy
- define network allowlist policy
- define secret scope policy
- define tool Components
- define Model Components
- define Agent Components
- implement inference logic
- expose Provider handles to Components
- make Wasmtime types part of the public API
- make ComponentEngine a Provider
- stabilize a generic dynamic invocation ABI
- implement hot reload
- implement cross-node Component execution

## Impact

Magnetar gains a real local WebAssembly Component execution path.

The Component Runtime becomes executable rather than only descriptive.

Future portable Components can be validated, linked, instantiated, invoked, and
interrupted through Magnetar.

The architectural boundary remains:

```text
Portable Component
        |
        v
Magnetar Component Runtime
        |
        v
ComponentEngine
        |
        v
Wasmtime adapter
```

Wasmtime is used as the first implementation but does not become the public
architecture.

This unlocks the next recadrage changes:

- Component artifact and trust model
- Component authority and scoping
- Component distribution contract
- Tachyon-provided Magnetar Components

and later functional domains:

- Model Components
- Observability Components
- Tool Components
- Agent Components