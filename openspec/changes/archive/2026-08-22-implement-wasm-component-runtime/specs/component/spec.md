## ADDED Requirements

### Requirement: Concrete WASM Component Engine

Magnetar SHALL provide at least one concrete WebAssembly Component Model engine
implementation behind the engine-neutral ComponentEngine boundary.

The initial implementation SHOULD use Wasmtime unless implementation evidence
requires another engine.

#### Scenario: Instantiate with concrete engine

Given a valid WASM Component artifact

And an authorized Link Plan

When the Runtime instantiates the Component

Then the concrete engine creates an executable Component Instance

And public Magnetar APIs remain engine-neutral.

---

### Requirement: Wasmtime Is an Implementation Detail

If Wasmtime is used, Wasmtime-native types SHALL remain private to the engine
adapter.

Canonical Magnetar Component APIs SHALL NOT expose concrete Wasmtime objects.

#### Scenario: Use Component Runtime API

Given application code uses Magnetar Component Runtime APIs

When Wasmtime is the concrete engine

Then application code does not require `wasmtime::Store`,
`wasmtime::component::Linker`, `wasmtime::component::Instance`, or
`wasmtime::Trap`.

---

### Requirement: Engine-Backed Component Preparation

The concrete Component Engine SHALL validate and prepare Component bytes before
instantiation.

Preparation MAY include engine parsing, validation, compilation, and
optimization.

#### Scenario: Invalid Component bytes

Given invalid Component bytes

When preparation is attempted

Then preparation fails with a stable Magnetar Component error.

---

### Requirement: Prepared Component Opaqueness

Prepared engine representation SHALL remain opaque outside the engine adapter.

Prepared state SHALL NOT cross WIT and SHALL NOT become a portable artifact.

#### Scenario: Cache prepared Component

Given the engine compiles a Component

When the prepared representation is cached

Then the cache remains internal to the Runtime and engine adapter.

---

### Requirement: WIT Import Inspection

The concrete engine integration SHALL support inspection or validation of WIT
imports required by a Component.

Required imports SHALL be matched against Runtime-owned Link Plans.

#### Scenario: Missing import

Given a Component imports an interface absent from the approved Link Plan

When instantiation is attempted

Then instantiation fails before execution.

---

### Requirement: WIT Export Inspection

The concrete engine integration SHALL support identifying Component exports
needed for invocation or validation.

An export SHALL NOT automatically become a globally available Capability.

#### Scenario: Component exports helper interface

Given a Component exports interface X

When the Component is registered

Then X is recorded as an export

But it is not globally linked to other Components without explicit Runtime
policy.

---

### Requirement: Runtime-Owned Link Plan Execution

The concrete engine adapter SHALL translate the Runtime-owned Link Plan into
engine-specific linker configuration.

Only approved imports SHALL be linked.

#### Scenario: Unauthorized import

Given a Component imports filesystem access

And the Runtime Link Plan does not authorize filesystem

When the adapter constructs the engine linker

Then filesystem is not linked.

---

### Requirement: No Ambient WASI

The concrete Component Engine SHALL NOT provide a broad ambient WASI
environment by default.

WASI interfaces SHALL be linked only when explicitly authorized.

#### Scenario: Component expects filesystem

Given a Component expects WASI filesystem access

And filesystem was not authorized

When the Component is linked or instantiated

Then the operation fails closed.

---

### Requirement: Capability Host Adapter

The concrete Component Runtime SHALL support host adapters that expose
Magnetar Runtime endpoints to Component imports.

A host adapter SHALL not expose native Provider or Device handles.

#### Scenario: Component imports test Capability

Given a Component imports an authorized Magnetar test Capability

When it invokes the host function

Then the call reaches the Runtime endpoint

And returns through the WASM Component boundary.

---

### Requirement: Capability Linking Does Not Resolve Provider

Linking a Provider-backed Capability import SHALL not select a concrete
Provider or Device.

#### Scenario: Link Compute import

Given a Component imports `magnetar:compute/run`

When the Component is instantiated

Then the import is linked to a Runtime Compute endpoint

And concrete Provider resolution is deferred until Compute work is submitted.

---

### Requirement: Async Host Call Scope

The concrete engine integration SHALL keep async host-call support internal to
the adapter boundary and SHALL NOT expose a concrete async runtime through
public Magnetar APIs.

The first implementation MAY support only synchronous unit-shaped host
adapters. If a linked Magnetar Capability requires asynchronous execution and
no typed async adapter exists, the adapter SHALL fail closed rather than
blocking a long-running Provider operation on an engine thread.

#### Scenario: Async Runtime endpoint

Given a linked host Capability completes asynchronously

When the Component invokes it

Then the engine adapter either coordinates completion through a typed Runtime
adapter

Or rejects the unsupported async host signature before execution.

---

### Requirement: Instance Store Isolation

Each Component Instance SHALL execute with isolated engine Store state.

#### Scenario: Two instances from one Component

Given one prepared Component definition

When the Runtime creates two Component Instances

Then each instance receives distinct engine execution state.

---

### Requirement: Engine Resource Tables Are Private

Engine resource table entries SHALL remain private implementation details.

They SHALL not become stable Magnetar resource identifiers.

The first implementation SHALL reject WIT resource imports unless an explicit
Runtime resource mapping exists for the linked host adapter.

#### Scenario: Engine creates resource entry

Given a Component call creates a WIT resource

When the engine stores the resource internally

Then the table entry is not exposed as a stable public Magnetar handle.

#### Scenario: Resource import lacks Runtime mapping

Given a Component imports a WIT resource

And no Runtime resource mapping exists for that resource type

When the Component is linked

Then instantiation fails closed.

---

### Requirement: Interruption Support

The concrete engine adapter SHALL support Runtime-requested interruption where
the engine can enforce it.

#### Scenario: Deadline expires

Given a Component invocation exceeds its configured deadline

When the Runtime requests interruption

Then the concrete engine attempts to interrupt execution

And the result is normalized to a Magnetar Component error.

---

### Requirement: Engine Trap Normalization

Engine traps SHALL be mapped into stable Magnetar Component trap errors.

#### Scenario: Component traps

Given Component execution traps inside the concrete engine

When the adapter reports the error

Then callers receive a Magnetar Component trap classification

And not the raw engine trap object.

---

### Requirement: Resource Limit Enforcement

The concrete engine adapter SHALL enforce configured resource limits where
supported.

If a required safety limit cannot be enforced, the adapter SHALL fail closed.

#### Scenario: Required memory limit unsupported

Given Runtime policy requires a Component memory limit

And the engine configuration cannot enforce that limit

When instantiation is attempted

Then instantiation fails rather than silently ignoring the policy.

---

### Requirement: Component Fixture Execution

The repository SHALL include at least one real WASM Component fixture that can
be prepared, linked, instantiated, and invoked by Magnetar tests.

#### Scenario: Execute fixture

Given the test fixture Component imports an authorized test Capability

When the end-to-end test runs

Then the Component invokes the Runtime host adapter successfully.

---

### Requirement: Unauthorized Import Fixture

The repository SHALL include a fixture or test proving that unauthorized imports
fail closed.

#### Scenario: Unauthorized filesystem import

Given a fixture Component requires filesystem access

And the Runtime does not authorize that import

When instantiation or linking occurs

Then the operation fails.

---

### Requirement: Trap Fixture

The repository SHALL include a fixture or test proving that Component traps are
normalized.

#### Scenario: Trap fixture executes

Given a fixture Component intentionally traps

When the Runtime invokes it

Then the error is reported as a stable Magnetar Component trap.

---

### Requirement: Multiple Instance Fixture

The repository SHALL include a fixture or test proving that multiple Component
instances created from one definition do not implicitly share mutable Store
state.

#### Scenario: Isolated instances

Given two instances are created from the same prepared Component

When one instance mutates its local state

Then the other instance does not observe that mutation.

---

### Requirement: Feature-Gated Engine Implementation

If the concrete engine is feature-gated, the repository SHALL provide a CI path
that enables and tests the feature.

#### Scenario: CI runs Component engine tests

Given the Wasmtime engine feature is optional

When CI validates the repository

Then at least one job enables the feature and runs end-to-end Component tests.

---

## MODIFIED Requirements

### Requirement: Component Discovery

The Runtime SHALL support discovery of local WebAssembly Component artifacts during
initialization or through explicit registration.

Discovered artifacts SHALL be validated and prepared by the Component Runtime
before instantiation.

Discovery alone SHALL NOT grant authority to execute imports.

#### Scenario: Discover Component file

Given a valid Component file exists in a configured directory

When the Runtime discovers it

Then the Component is known to the Runtime

But it is not instantiated with unauthorized imports.

---

### Requirement: Component Isolation

Components SHALL execute through a real WebAssembly Component Model engine while
remaining isolated from native Providers, Devices, and engine-native handles.

#### Scenario: Execute with native Provider present

Given a CUDA Provider is registered in the Runtime

And a Component imports an authorized Runtime Capability

When the Component executes

Then it cannot access CUDA-native handles unless a future explicit portable
contract permits an opaque resource.

---

### Requirement: Component Lifecycle

The Runtime SHALL manage lifecycle for engine-backed Component definitions and
instances.

A Component definition SHALL be validated and prepared before instantiation.

A Component Instance SHALL be destroyed before its engine Store state is
released.

#### Scenario: Runtime shutdown

Given an engine-backed Component Instance exists

When Runtime shutdown occurs

Then the Runtime prevents new invocations and destroys the instance according
to Runtime policy.
