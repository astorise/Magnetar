## ADDED Requirements

### Requirement: Runtime Uses ComponentEngine for WASM Execution

The Runtime SHALL execute WebAssembly Components through the engine-neutral
ComponentEngine boundary.

The Runtime SHALL NOT directly expose concrete engine objects as public API.

#### Scenario: Prepare Component

Given Runtime receives Component bytes

When preparation begins

Then Runtime delegates engine-specific preparation to ComponentEngine

And stores only Magnetar-owned public state outside the adapter.

---

### Requirement: Runtime Builds Linker from Link Plan

The Runtime SHALL use its approved Component Link Plan as the sole source of
truth for constructing the concrete engine linker.

#### Scenario: Link authorized import

Given a Component imports interface X

And Runtime policy authorizes X

When the engine linker is constructed

Then X is linked through the Runtime-approved endpoint.

---

### Requirement: Runtime Denies Imports Absent from Link Plan

An import absent from the approved Link Plan SHALL be unavailable to the
Component.

#### Scenario: Import not authorized

Given a Component imports interface Y

And Y is absent from the approved Link Plan

When instantiation is attempted

Then the Runtime fails the operation rather than linking Y.

---

### Requirement: Runtime Hosts Capability Endpoints

The Runtime SHALL provide host-call endpoints for authorized Magnetar
Capabilities imported by Components.

These endpoints SHALL call Runtime services.

They SHALL NOT expose native Provider or Device handles.

#### Scenario: Component calls host Capability

Given a Component invokes an authorized host Capability

When the host adapter executes

Then control enters Runtime-managed code

And any Provider-backed work follows normal Runtime resolution.

---

### Requirement: Runtime Preserves Provider Resolution Boundary

Instantiating a Component SHALL NOT select a concrete Provider or Device merely
because one of its imports is Provider-backed.

#### Scenario: Instantiate Compute Component

Given a Component imports Compute

When the Component is instantiated

Then Runtime links a Compute endpoint

But Provider and Device selection occur only when Compute work is requested.

---

### Requirement: Runtime Tracks Engine-Backed Instances

The Runtime SHALL associate every engine-backed Component Instance with
Runtime-owned identity and lifecycle state.

#### Scenario: Instance created

Given ComponentEngine creates a new executable instance

When instantiation succeeds

Then Runtime records its ComponentInstanceId and lifecycle state.

---

### Requirement: Runtime Owns Store Lifetime

Engine Store state SHALL be associated with Component Instance lifetime.

The Runtime SHALL ensure that Store state is released when the instance is
destroyed.

#### Scenario: Destroy instance

Given a Component Instance is destroyed

When destruction completes

Then engine Store state is no longer usable through Runtime invocation APIs.

---

### Requirement: Runtime Enforces No Ambient Authority

Runtime SHALL configure the concrete engine so that Components receive no
ambient authority.

This includes at least:

- filesystem
- network
- environment variables
- process execution
- secrets
- sockets
- broad WASI environment

unless explicitly authorized.

#### Scenario: Component attempts environment access

Given no environment interface is authorized

When Component linking occurs

Then environment access is not provided by the Runtime.

---

### Requirement: Runtime Applies Component Resource Policy

Runtime SHALL translate Component resource policy into concrete engine
configuration where feasible.

If policy cannot be enforced, Runtime SHALL fail closed.

#### Scenario: Required deadline enforcement unavailable

Given Runtime policy requires interruptible execution

And the selected engine configuration cannot support interruption

When the Component is prepared or instantiated for that policy

Then Runtime rejects the configuration.

---

### Requirement: Runtime Normalizes Engine Errors

Runtime SHALL map engine-specific errors into stable Magnetar Component errors.

#### Scenario: Engine returns Wasmtime error

Given the concrete engine reports a Wasmtime-specific failure

When the error crosses the Component Runtime boundary

Then Runtime exposes a stable Magnetar error classification.

---

### Requirement: Runtime Separates Component and Provider Failure

Runtime SHALL preserve the distinction between Component Engine failures,
Component traps, Runtime host-call failures, and Provider execution failures.

#### Scenario: Provider fails during host call

Given a Component successfully invokes a Runtime host Capability

And the selected Provider fails during native execution

When the error is returned

Then Runtime reports the Provider failure through the relevant Capability error

And does not classify the engine itself as failed.

---

### Requirement: Runtime Observes Engine Operations

Runtime SHALL support structured observations for engine-backed Component
operations.

Observations SHALL remain non-authoritative.

#### Scenario: Invocation observed

Given a Component invocation completes

When Runtime emits observability data

Then the observation may include Component instance identity and duration

But does not alter the invocation result.

---

### Requirement: Runtime CI Validates Real Component Execution

Repository CI SHALL include validation for the concrete Component engine.

At least one CI job SHALL prepare, instantiate, and invoke a real WASM
Component fixture.

#### Scenario: Component engine regression

Given a change breaks concrete Component invocation

When CI executes Component Runtime tests

Then the workflow fails.