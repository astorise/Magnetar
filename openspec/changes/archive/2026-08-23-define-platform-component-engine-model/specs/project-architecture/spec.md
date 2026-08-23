## ADDED Requirements

### Requirement: Component Engine Is Platform-Aware

Magnetar SHALL support platform-specific Component Engine implementations behind
the Component Runtime boundary.

#### Scenario: Native build

Given Magnetar is built for a native target

When a compatible native Component Engine is enabled

Then Runtime may use the native engine implementation.

#### Scenario: Browser build

Given Magnetar is built for `wasm32`

When Component execution is enabled

Then Runtime uses a web-compatible engine or returns a structured unsupported
engine error.

---

### Requirement: Wasmtime Is Native Implementation Not Universal Architecture

Wasmtime SHALL be treated as an optional native Component Engine
implementation.

Wasmtime SHALL NOT be required for browser targets.

#### Scenario: wasm32 target

Given Magnetar is compiled for `wasm32-unknown-unknown`

When the Runtime is checked

Then Wasmtime-specific code is not required to compile.

---

### Requirement: Browser Engine Is Separate Implementation

Browser targets SHALL use a separate Component Engine implementation or adapter.

The browser engine MAY use browser WebAssembly APIs and JavaScript-mediated host
bindings.

#### Scenario: Browser Component execution

Given a Component is prepared in a browser target

When Runtime builds host bindings

Then bindings are produced through the browser-compatible engine

And not through Wasmtime.

---

### Requirement: Engine Profile Compatibility

Component Engine implementations SHALL declare their platform profile and
capabilities.

Runtime SHALL reject Components requiring unavailable engine profiles or
features.

#### Scenario: Native-only Component on web

Given a Component requires native engine features unavailable in browser

When loaded on a browser target

Then Runtime rejects it before preparation.

---

### Requirement: No Ambient Authority Across Engines

No Component Engine implementation SHALL grant ambient filesystem, network,
process, secret, Git, workspace, or broad WASI authority to Magnetar Components.

#### Scenario: Browser APIs available

Given browser APIs exist

When a Magnetar Component is instantiated

Then those APIs are not linked unless explicitly authorized by Magnetar Runtime
policy.

---

### Requirement: Native Provider Loading Is Not Browser Requirement

Dynamic native Provider loading SHALL not be required for browser builds.

#### Scenario: Web build

Given Magnetar is compiled for browser target

When Provider loading features are evaluated

Then native dynamic Provider loading is unavailable or excluded.
