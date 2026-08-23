# Define Platform Component Engine Model

## Why

Magnetar defines a Component Runtime boundary and has a concrete native
Wasmtime-based implementation path.

However, Wasmtime is not a universal platform abstraction.

A native server target and a browser `wasm32-unknown-unknown` target have
different execution constraints.

A browser build cannot assume:

- native dynamic libraries
- native threads in the same way as server runtimes
- direct filesystem access
- process execution
- native Provider loading
- Wasmtime availability
- OS-level memory mapping
- OS-level interruption primitives
- unrestricted WASI
- native pinned memory
- native GPU Provider loading

Therefore, the Component Engine model must become platform-aware.

The problem is not only that a given implementation file exists.

The architectural problem is:

```text
ComponentEngine must not be implicitly Wasmtime-only.
```

Magnetar needs a model where:

```text
native targets
    use a native Component Engine such as Wasmtime

browser targets
    use a web-compatible Component Engine adapter

Runtime policy
    selects an engine based on target platform and enabled features
```

This change defines that platform-specific Component Engine model.

## What Changes

This change introduces a platform-aware Component Engine architecture.

The Component Runtime remains the stable Magnetar abstraction.

Concrete engines become platform-specific implementations behind that boundary.

Conceptual model:

```text
Component Runtime
        |
        v
ComponentEngine trait / abstraction
        |
        +-- NativeComponentEngine
        |       |
        |       +-- WasmtimeComponentEngine
        |
        +-- WebComponentEngine
                |
                +-- Browser-hosted WASM / JS adapter
```

The exact Rust type names are implementation-defined.

The architectural requirement is that Magnetar Runtime does not become
Wasmtime-only.

## Native Component Engine

Native targets MAY use Wasmtime as the concrete Component Engine.

Native targets SHOULD gate Wasmtime-specific implementation behind:

```rust
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime-component-engine"))]
```

or an equivalent target-aware configuration.

Native Component Engine responsibilities MAY include:

- Component validation
- Component Model support
- compilation/preparation
- instantiation
- linker construction
- host call integration
- async host calls
- trap normalization
- interruption where supported
- resource limits where supported
- no ambient authority
- no broad WASI unless explicitly linked

## Web Component Engine

Browser targets SHALL use a web-compatible Component Engine.

A web engine MAY be implemented through a module such as:

```text
component_web.rs
```

or:

```text
component/web.rs
```

The web engine MAY use:

- `wasm-bindgen`
- `js-sys`
- `web-sys`
- browser WebAssembly APIs
- JavaScript-mediated host calls
- browser-specific resource policy
- browser-specific memory policy

The web engine SHALL be selected through target-aware configuration, such as:

```rust
#[cfg(target_arch = "wasm32")]
```

The web engine SHALL NOT import or require Wasmtime.

## Component Engine Profiles

Magnetar SHALL define Component Engine profiles.

Initial profiles SHOULD include:

```text
component-engine-native
component-engine-web
component-engine-test
```

A profile describes which Component Engine features are available on a platform.

A Component Artifact or Runtime configuration MAY declare required engine
features.

The Runtime SHALL reject or refuse to prepare Components whose required engine
features are unavailable on the current platform.

## Native Profile

The native profile MAY support:

- native Component Model execution
- Wasmtime-based preparation
- async host calls
- Runtime Link Plans
- interruption where available
- resource limits where available
- native observability integration
- native Provider-backed Capability endpoints
- native memory manager integration
- dynamic Provider loading where policy allows

The native profile SHALL still enforce:

- no ambient authority
- inference-scoped Component authority
- no direct Provider/Device selection by Components
- no raw native handle exposure

## Web Profile

The web profile SHALL represent browser constraints explicitly.

The web profile MAY support:

- browser WebAssembly execution
- JavaScript-mediated host calls
- browser-managed module instantiation
- Runtime Link Plan translation to JS host bindings
- browser-compatible observability
- browser-compatible memory manager integration
- inference-scoped Component authority

The web profile SHALL NOT assume:

- Wasmtime
- native dynamic Provider loading
- native filesystem access
- process execution
- native pinned memory
- OS-level signals
- native threads unless explicitly available
- unrestricted WASI
- arbitrary network authority inside Magnetar Components

## Test Profile

The test profile MAY use a mock ComponentEngine.

The test profile SHALL be able to simulate:

- preparation success
- preparation failure
- instantiation success
- instantiation failure
- missing imports
- unauthorized imports
- trap
- interruption
- resource-limit failure
- destruction

The test profile exists for contract and failure testing.

It SHALL not imply that the mock engine is production-capable.

## Engine Selection

Runtime SHALL select a Component Engine based on:

- target architecture
- enabled features
- Runtime configuration
- Component Artifact requirements
- platform capabilities
- policy

Selection SHALL fail closed when no compatible engine exists.

A native build MAY select Wasmtime.

A browser build SHALL select the web engine or fail with a clear unsupported
platform error.

A browser build SHALL NOT accidentally pull in Wasmtime-only dependencies.

## Build Gating

Wasmtime-specific code SHALL be target-gated.

Browser-specific code SHALL be target-gated.

Recommended source organization:

```text
magnetar-runtime/src/
├── component.rs
├── component_wasmtime.rs
└── component_web.rs
```

or:

```text
magnetar-runtime/src/component/
├── mod.rs
├── engine.rs
├── wasmtime.rs
└── web.rs
```

The exact layout is implementation-defined.

The important rule is:

```text
native-only engine code must not compile into wasm32 browser builds
web-only engine code must not compile into native builds unless explicitly used
```

## Cargo Features

Cargo features SHALL not be the only gating mechanism.

Target architecture must also be considered.

For example:

```text
feature = wasmtime-component-engine
target = native

feature = web-component-engine
target = wasm32
```

A feature combination that is impossible for the target SHALL fail clearly or be
disabled by cfg.

## Runtime API Stability

The public Component Runtime API SHALL remain stable across platform engines.

Platform differences SHALL be represented through:

- engine capabilities
- engine profile
- unsupported feature errors
- Component Artifact compatibility validation
- Runtime diagnostics

The Runtime SHALL NOT expose Wasmtime-specific types in public portable APIs.

The Runtime SHALL NOT expose browser JavaScript objects in public portable APIs
unless they are behind explicit web-only APIs.

## Link Plan Translation

Runtime Link Plans remain Runtime-owned.

A native engine translates Link Plans into native engine linker bindings.

A web engine translates Link Plans into browser or JavaScript host bindings.

The Component SHALL still receive only authorized inference-scoped imports.

The web engine SHALL not use JavaScript host bindings as a bypass around
Magnetar authority validation.

## WASI Policy

No platform engine SHALL provide ambient WASI by default.

Native Wasmtime MAY support controlled WASI when explicitly linked.

Browser engines MAY expose browser host functions only when explicitly linked.

WASI or browser APIs SHALL not grant broad filesystem, network, secrets,
workspace, Git, or process authority to Magnetar Components.

## Provider Boundary On Web

Browser targets SHALL NOT assume native Provider loading.

Dynamic native Provider loading is not available in browser builds.

A browser build MAY use:

- pure WASM inference Providers in the future
- WebGPU-backed Providers in the future
- JavaScript-mediated inference adapters
- browser-native APIs exposed through controlled Runtime endpoints

Those are future Provider models.

This change only ensures the Component Engine model does not require native
Provider loading.

## Memory Boundary On Web

Browser engines SHALL integrate with Memory Manager through browser-compatible
memory placement.

Browser targets SHALL not assume native pinned memory, native mmap, or native
allocator behavior.

Browser memory constraints SHALL be surfaced through Memory Manager capability
and error reporting.

## Artifact Compatibility

Component Artifacts MAY declare required engine features.

Examples:

```text
requires engine profile = component-engine-native
requires engine feature = component-model
requires engine feature = async-host-calls
requires engine feature = browser-compatible
```

Runtime SHALL reject incompatible Components before preparation.

## Diagnostics

Platform engine failures SHALL produce structured diagnostics.

Diagnostic categories SHOULD include:

- no compatible engine
- engine feature unavailable
- engine profile mismatch
- target unsupported
- Wasmtime unavailable
- browser engine unavailable
- import linking unsupported
- trap
- interruption unsupported
- resource limits unsupported
- host binding failed

Diagnostics SHALL not expose unsafe native handles or private browser objects.

## Observability

Component Engine operations SHOULD emit observations for:

- engine selected
- engine rejected
- platform unsupported
- preparation start
- preparation success
- preparation failure
- instantiation success
- instantiation failure
- Link Plan translated
- host binding failure
- trap
- interruption
- destruction

Observability SHALL not alter engine selection or execution correctness.

## Non-Goals

This change does not:

- implement a complete browser inference runtime
- implement WebGPU Provider
- implement JavaScript Provider ABI
- implement model loading
- implement tokenizer contract
- implement generation
- implement browser UI
- define browser persistence
- define browser networking
- grant browser filesystem authority
- grant browser network authority
- require Wasmtime on wasm32
- remove Wasmtime support on native targets
- define full WASI policy
- define service-worker execution
- define remote execution

## Impact

Magnetar gains a platform-aware Component Engine model.

Native builds can continue using Wasmtime.

Browser builds gain an architectural path through a web-compatible engine.

The Runtime remains stable because Components target Magnetar Capability
contracts, not a specific engine implementation.

The architecture becomes:

```text
Component Runtime
        |
        +-- native target
        |       |
        |       +-- WasmtimeComponentEngine
        |
        +-- wasm32 browser target
                |
                +-- WebComponentEngine
```

This prevents the native server path from blocking Magnetar Web.