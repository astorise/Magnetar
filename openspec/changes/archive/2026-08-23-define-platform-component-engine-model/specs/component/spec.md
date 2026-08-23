## ADDED Requirements

### Requirement: Component Engine Profiles

Component Engine implementations SHALL declare an engine profile.

Initial profiles SHOULD include:

- component-engine-native
- component-engine-web
- component-engine-test

#### Scenario: Engine reports profile

Given Runtime initializes a Component Engine

When it reads engine capabilities

Then the engine reports its profile.

---

### Requirement: Native Component Engine Profile

The native Component Engine profile SHALL represent native Component execution
capabilities and MAY support Wasmtime-based Component execution.

Native engine support SHALL be target-gated and feature-gated.

#### Scenario: Native Wasmtime engine

Given a native build enables the Wasmtime Component Engine feature

When Runtime selects an engine

Then Wasmtime may be selected if compatible.

---

### Requirement: Web Component Engine Profile

The web Component Engine profile SHALL be compatible with browser targets.

It SHALL NOT depend on Wasmtime.

#### Scenario: Web engine selected

Given Magnetar is built for `wasm32`

When Runtime selects a Component Engine

Then a web-compatible Component Engine is selected if available.

---

### Requirement: ComponentEngineCapabilities

Each Component Engine SHALL expose capabilities.

Capabilities MAY include:

- component model support
- async host calls
- interruption support
- resource limits
- controlled WASI support
- browser compatibility
- native Provider endpoint support
- JS-mediated host call support

#### Scenario: Feature unavailable

Given a Component requires interruption support

And the selected engine does not support interruption

When Runtime validates the Component against engine capabilities

Then validation fails before preparation.

---

### Requirement: Component Artifact Engine Requirements

A Component Artifact SHALL be allowed to declare required Component Engine
profile or features.

Runtime SHALL evaluate these requirements before preparation.

#### Scenario: Artifact requires web engine

Given a Component Artifact declares it requires `component-engine-web`

When loaded on a native-only Runtime

Then Runtime rejects it or reports no compatible engine.

---

### Requirement: Engine Selection Fails Closed

If no compatible Component Engine exists, Runtime SHALL fail closed.

#### Scenario: No engine

Given Component execution is requested

And no engine supports the current target and artifact requirements

When Runtime selects an engine

Then Runtime returns a structured no-compatible-engine error.

---

### Requirement: Platform-Neutral Component Runtime API

The public Component Runtime API SHALL remain platform-neutral.

Wasmtime-specific types and browser JavaScript objects SHALL not leak into the
portable Component Runtime API.

#### Scenario: Public API inspection

Given a caller uses the Component Runtime API

When they prepare or instantiate a Component

Then they do not need to name Wasmtime or JavaScript engine-specific types.

---

### Requirement: Link Plan Translation Is Engine-Specific

Runtime Link Plans SHALL remain platform-neutral.

Each Component Engine SHALL translate Link Plans into its own host binding
mechanism.

#### Scenario: Web host binding

Given a web Component Engine receives a Link Plan

When it creates host bindings

Then it uses browser-compatible bindings

And preserves Runtime authorization decisions.

---

### Requirement: No Ambient WASI By Engine

Component Engines SHALL NOT provide ambient WASI or broad host APIs by default.

#### Scenario: Native engine supports WASI

Given Wasmtime can support WASI

When Magnetar instantiates an inference Component

Then broad WASI is not linked unless explicitly authorized.

#### Scenario: Browser engine can call JS

Given browser JavaScript APIs exist

When Magnetar instantiates an inference Component

Then arbitrary JS APIs are not linked unless explicitly authorized.
