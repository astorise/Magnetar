## ADDED Requirements

### Requirement: Runtime Selects Component Engine By Platform

Runtime SHALL select Component Engine implementation using platform, feature,
configuration, and Component Artifact requirements.

#### Scenario: Native target

Given Runtime runs on a native target

And the Wasmtime feature is enabled

When a compatible Component is prepared

Then Runtime may select the Wasmtime Component Engine.

#### Scenario: Browser target

Given Runtime runs on `wasm32`

And a web Component Engine is available

When a compatible Component is prepared

Then Runtime selects the web Component Engine.

---

### Requirement: Runtime Does Not Require Wasmtime On Browser Targets

Runtime SHALL NOT require Wasmtime for browser targets.

#### Scenario: wasm32 compile check

Given the target is `wasm32-unknown-unknown`

When Runtime is compiled or checked

Then Wasmtime-specific modules are not required.

---

### Requirement: Runtime Rejects Incompatible Engine Requirements

Runtime SHALL reject Component Artifacts whose engine requirements cannot be
satisfied on the current platform.

#### Scenario: Component requires native resource limits

Given a Component requires a native-only resource limit feature

And Runtime is running on a web engine without that feature

When validation runs

Then Runtime rejects the Component before preparation.

---

### Requirement: Runtime Keeps Engine Details Internal

Runtime SHALL keep concrete engine details internal.

Public Runtime APIs SHALL not expose Wasmtime-specific or browser-specific
engine internals.

#### Scenario: Runtime caller prepares Component

Given a caller prepares a Component through Runtime

When the concrete engine is selected internally

Then the caller receives platform-neutral Runtime results.

---

### Requirement: Runtime Translates Link Plans Through Selected Engine

Runtime SHALL provide the selected Component Engine with a Runtime-owned Link
Plan.

The selected engine SHALL translate that plan into platform-specific host
bindings.

#### Scenario: Web Link Plan

Given Runtime selects the web engine

When the Component requires Compute import

Then Runtime-authorized Compute host binding is translated into a
browser-compatible binding.

---

### Requirement: Runtime Enforces Authority Before Engine Binding

Runtime SHALL validate authority before any Component Engine binds host
functions.

#### Scenario: Browser JS binding

Given a Component requests forbidden network authority

When Runtime validates the Component

Then no browser JS network binding is created.

---

### Requirement: Runtime Reports Platform Engine Errors

Runtime SHALL normalize platform engine failures into structured Component
errors.

#### Scenario: Wasmtime feature disabled

Given a native Runtime has no Component Engine enabled

When Component preparation is requested

Then Runtime returns a no-compatible-engine error.

---

### Requirement: Runtime Observes Engine Selection

Runtime SHALL define observations for Component Engine selection and rejection.

#### Scenario: Engine rejected

Given a Component requires web profile

And only a native engine is available

When Runtime rejects the engine

Then Runtime may emit an engine-profile-mismatch observation.
