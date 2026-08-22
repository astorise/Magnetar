## MODIFIED Requirements

### Requirement: Component Contracts

Every Component SHALL declare its portable dependencies and exports through WIT
interfaces.

Component imports SHALL identify required interfaces rather than concrete native
implementations or Component instance names.

The Runtime SHALL validate required imports before instantiation.

#### Scenario: Validate contracts

Given a Component imports one or more WIT interfaces

When the Runtime prepares the Component for instantiation

Then every required interface is validated for compatibility and authorization

And unresolved mandatory imports prevent instantiation.

---

### Requirement: Component Isolation

Components SHALL execute independently from hardware implementations and from
engine-native Runtime objects.

Components SHALL NOT receive:

- Provider handles
- Device handles
- native pointers
- engine Store handles
- engine Linker handles
- engine ResourceTable handles
- native queue or stream handles

through portable contracts.

#### Scenario: Execute on different Runtime environments

Given the same compatible Component

And two Runtime environments with different native Providers

When the Component executes

Then no Component modification is required

And the Component does not observe the native implementation handles.

---

### Requirement: Component Lifecycle

The Runtime SHALL manage Component definition and Component instance lifecycle.

Generic Components SHALL NOT be required to implement universal `start` or
`stop` exports.

A successfully instantiated Component Instance SHALL become available according
to its exported WIT contracts.

Application-specific lifecycle behavior MAY be defined through explicit WIT
interfaces.

#### Scenario: Instantiate Component without start export

Given a valid Component has no generic `start` export

When the Runtime validates, prepares, links, and instantiates it successfully

Then the Component Instance becomes available for invocation.

#### Scenario: Runtime shutdown

Given one or more Component Instances exist

When the Runtime shuts down

Then new invocation is prevented

And the Runtime releases each instance according to Component Runtime lifecycle
policy

Without requiring an implicit portable `stop` function.

---

### Requirement: Dependency Resolution

Component dependencies SHALL be expressed through WIT imports.

Components SHALL NOT require direct dependency on another Component's logical
name as the canonical dependency mechanism.

The Runtime SHALL resolve and authorize required interfaces before
instantiation.

#### Scenario: Resolve Capability dependency

Given a Component imports `magnetar:compute/run`

When the Runtime constructs its Link Plan

Then the import is linked to an authorized Runtime Compute Capability endpoint

And the Component does not name the Component or Provider implementing the
underlying behavior.

---

## ADDED Requirements

### Requirement: Component Runtime

Magnetar SHALL provide an engine-independent Component Runtime layer.

The Component Runtime SHALL own Magnetar-specific Component policy and
orchestration.

#### Scenario: Instantiate portable Component

Given a validated Component definition

When instantiation is requested

Then the Component Runtime constructs the authorized Link Plan

And delegates engine-specific instantiation to ComponentEngine.

---

### Requirement: Component Engine Boundary

Magnetar SHALL isolate concrete WebAssembly engine behavior behind a
ComponentEngine abstraction.

ComponentEngine SHALL own engine-specific preparation, instantiation, execution,
interruption, and cleanup mechanics.

#### Scenario: Replace engine implementation

Given two Component Engines implement the required Magnetar engine boundary

When one implementation is replaced by another

Then portable Component contracts and Magnetar Capability contracts do not need
to change.

---

### Requirement: Component Engine Is Not a Provider

ComponentEngine SHALL NOT be modeled as a Provider.

Engine implementation capabilities SHALL NOT be registered as Magnetar
Capabilities merely because the engine supports them.

#### Scenario: Engine supports interruption

Given the concrete WASM engine supports execution interruption

When that implementation feature is recorded

Then it is treated as Component Engine capability

And not as a Provider Capability exposed to Components.

---

### Requirement: Concrete Engine Isolation

Canonical Magnetar Component APIs SHALL NOT expose concrete engine types.

#### Scenario: Wasmtime adapter

Given the future engine implementation uses Wasmtime

When Runtime code outside the engine adapter manages a Component

Then it does not require `wasmtime::Store`, `wasmtime::component::Linker`, or
equivalent Wasmtime-native objects.

---

### Requirement: Component Definition and Instance Are Distinct

Magnetar SHALL distinguish Component definition identity from Component instance
identity.

One Component definition MAY create multiple isolated instances.

#### Scenario: Instantiate twice

Given one prepared Component definition

When the Runtime creates two instances

Then each instance has distinct Runtime identity

And mutable instance state is not implicitly shared.

---

### Requirement: Prepared Component Is Engine Opaque

A Component Engine SHALL keep any prepared or compiled representation private.

It MAY maintain such a representation internally.

That representation SHALL remain opaque outside the engine adapter and SHALL
NOT be portable WIT state.

#### Scenario: Prepare Component

Given valid Component bytes

When ComponentEngine prepares them

Then the resulting prepared representation may be cached internally

But a Component cannot access or serialize the engine-native representation as a
Magnetar portable resource.

---

### Requirement: Instance Store Isolation

Every Component Instance SHALL execute with isolated engine-managed mutable
state unless explicit sharing has been defined by a future contract.

#### Scenario: Two Component instances

Given two instances originate from the same definition

When one modifies its Component-local mutable state

Then the other instance does not observe that state merely because the
definition is shared.

---

### Requirement: Runtime-Owned Link Plan

The Runtime SHALL construct an immutable Link Plan for Component instantiation.

The Link Plan SHALL describe which imports are satisfied and authorized.

Component code SHALL NOT modify its own Link Plan after instantiation.

#### Scenario: Link Compute Capability

Given a Component imports `magnetar:compute/run`

And the import is authorized

When its Link Plan is created

Then the Runtime provides a Compute Capability endpoint for that import.

---

### Requirement: Capability Linking Does Not Pin Provider

Linking a Magnetar Capability SHALL NOT by itself select a permanent Provider or
Device for the Component.

Provider and Device Resolution SHALL continue according to Capability
requirements, Resolution Policy, Resource Affinity, and execution state.

#### Scenario: Component imports Compute

Given a Component is linked to `magnetar:compute/run`

And no Provider-owned state exists yet

When separate Compute operations execute

Then each operation may be resolved according to Runtime policy

And linking itself does not pin the Component to CUDA, CPU, or another Provider.

---

### Requirement: No Automatic Component Export Linking

A Component export SHALL NOT automatically become a globally available import
for other Components.

Explicit composition requires Runtime policy or a future Component composition
contract.

#### Scenario: Two matching interfaces

Given Component A exports interface X

And Component B imports interface X

When both are registered

Then the Runtime does not automatically connect B to A solely because the
interfaces match.

---

### Requirement: No Component Name Service Location

A Component logical name SHALL NOT serve as the canonical mechanism for locating
a dependency implementation.

#### Scenario: Replace implementation

Given a Component requires interface X

And the Runtime changes which authorized endpoint satisfies X

When the requiring Component executes

Then its WIT contract remains unchanged.

---

## ADDED Requirements

### Requirement: No Ambient Component Authority

A Component SHALL receive only interfaces explicitly linked by the Runtime.

An interface that is not linked SHALL be unavailable.

#### Scenario: Component attempts undeclared host access

Given filesystem access has not been linked to a Component

When the Component attempts to use filesystem functionality

Then the functionality is unavailable.

---

### Requirement: WASI Is Explicit

WASI interfaces SHALL NOT be automatically granted as a broad default
environment.

Each available WASI interface SHALL require explicit Runtime linking and policy.

#### Scenario: Component does not require WASI filesystem

Given a Component imports only Magnetar Compute

When it is instantiated

Then filesystem access is not implicitly available.

---

## ADDED Requirements

### Requirement: Contract-Specific Invocation

Component invocation SHALL use WIT contract-specific interfaces.

Magnetar SHALL NOT require a generic string-based dynamically typed invocation
ABI as its canonical public Component API.

#### Scenario: Invoke typed export

Given a Component exports a known WIT interface

When the Runtime invokes the export

Then invocation may use generated or contract-specific typed bindings.

---

### Requirement: Asynchronous Host Capability Support

The Component Runtime boundary SHALL permit host Capability operations that
complete asynchronously.

#### Scenario: Component submits Compute

Given a Component invokes a Compute Capability operation

And the Provider performs asynchronous native execution

When the host call is handled

Then the Component Runtime architecture does not require blocking a native
execution thread solely because the call crossed the WASM boundary.

---

## ADDED Requirements

### Requirement: Component Interruption

The Runtime SHALL support requesting interruption of Component execution.

Interruption MAY be triggered by:

- cancellation
- shutdown
- deadline
- resource policy
- administrative termination

#### Scenario: Invocation deadline expires

Given a Component invocation exceeds its Runtime deadline

When interruption is requested

Then the Component Engine attempts to interrupt execution

And the result is represented using stable Magnetar Component semantics.

---

### Requirement: Engine Interruption Mechanisms Are Private

Concrete interruption mechanisms such as fuel or epoch interruption SHALL remain
Component Engine implementation details.

#### Scenario: Wasmtime uses epoch interruption

Given the concrete engine implements deadlines using epoch interruption

When a Component is interrupted

Then the public Magnetar error describes interruption

And does not require the caller to understand engine epochs.

---

### Requirement: Component Trap Normalization

Engine-specific Component traps SHALL be mapped to stable Magnetar Component
errors.

#### Scenario: WebAssembly traps

Given Component execution traps

When the Component Engine reports the failure

Then the Runtime returns a stable Component trap classification

And MAY attach redacted diagnostics

Without exposing the concrete engine Trap type.

---

### Requirement: Component Failure Is Not Provider Failure

A Component trap SHALL NOT automatically mark its resolved Provider or Device as
failed.

#### Scenario: Component executes invalid WASM logic

Given a Component traps before invoking native Compute

When the invocation fails

Then the Component invocation fails

And Provider health remains unchanged.

---

### Requirement: Provider Failure Is Not Engine Failure

A Provider execution failure SHALL NOT automatically mark ComponentEngine as
failed.

#### Scenario: GPU execution fails

Given a Component invokes Compute successfully through the WASM engine

And the selected GPU Provider later fails execution

When the error is returned

Then the Provider failure is reported through Compute semantics

And the Component Engine itself is not classified as failed solely for that
reason.

---

### Requirement: Component and Provider Cancellation Are Distinct

Interrupting a Component SHALL NOT imply that already-submitted Provider work
has been cancelled.

Provider work SHALL follow its own cancellation contract.

#### Scenario: Component cancelled after Compute submission

Given a Component has submitted native Compute work

When the Component invocation is interrupted

Then the Runtime applies Provider cancellation semantics where appropriate

And does not assume that terminating WASM execution automatically cancels
native work.

---

## ADDED Requirements

### Requirement: Instance Resource Ownership

WIT resources associated with a Component Instance SHALL have explicit instance
ownership.

A Component SHALL NOT forge access to a resource owned exclusively by another
instance.

#### Scenario: Cross-instance resource forgery

Given Component Instance A owns an instance-local WIT resource

When Instance B presents an unrelated or forged resource identity

Then the Runtime rejects access.

---

### Requirement: Engine Resource Handles Are Private

Engine resource-table handles SHALL NOT become stable Magnetar portable resource
identifiers.

#### Scenario: Engine allocates resource table entry

Given an engine creates an internal resource-table index

When the resource crosses a Magnetar WIT contract

Then the index remains an engine implementation detail.

---

### Requirement: Instance Destruction Releases Instance Resources

Destroying a Component Instance SHALL release engine resources owned exclusively
by that instance.

Independently owned Runtime resources SHALL follow their own lifecycle.

#### Scenario: Destroy Component instance

Given an instance owns Component-local resources

And separately references a Runtime-managed tensor resource

When the instance is destroyed

Then Component-local engine resources are released

And tensor lifetime follows the Runtime's resource ownership rules.

---

## ADDED Requirements

### Requirement: Component Resource Limits

The Component Runtime SHALL support semantic execution limits.

Limits MAY include:

- memory ceiling
- execution deadline
- maximum concurrent invocations
- Runtime instance-count limits
- engine execution budget

#### Scenario: Required memory limit

Given Runtime policy requires a Component memory ceiling

When the engine cannot enforce the required safety property

Then instantiation fails closed rather than silently ignoring the policy.

---

### Requirement: Engine Limit Mechanisms Are Not Portable Contracts

Engine-specific control mechanisms SHALL NOT become portable Component
contracts.

These mechanisms include fuel, epoch counters, Store limiters, pooling
allocators, and equivalent engine details.

#### Scenario: Engine implementation changes

Given a different Component Engine uses another resource-limit mechanism

When it satisfies the same Runtime policy

Then Component WIT contracts do not change.

---

### Requirement: Safe Instance Concurrency

The Runtime SHALL respect Component Engine Store and instance concurrency
requirements.

It SHALL NOT concurrently mutate an instance's engine state in a way prohibited
by the engine.

#### Scenario: Two calls target same instance

Given the selected engine does not support concurrent mutable entry into one
Store

When two calls target the same Component Instance

Then the Component Runtime serializes or otherwise safely coordinates them.

---

### Requirement: Independent Instances May Execute Concurrently

Independent Component Instances SHALL remain eligible for concurrent execution
when Runtime and engine policy permit.

#### Scenario: Two isolated model Components

Given two independent Component Instances

When both are ready

Then the Runtime may execute them concurrently without sharing mutable Store
state.

---

## ADDED Requirements

### Requirement: Component Runtime Observability

The Component Runtime SHALL support structured Runtime observations for
important lifecycle and execution events.

Observations MAY include:

- definition identity
- instance identity
- preparation
- instantiation
- invocation
- interruption
- trap
- resource-limit violation
- destruction

#### Scenario: Component traps

Given a Component invocation traps

When Runtime observability records the failure

Then the observation identifies the relevant Component instance and stable trap
category

Without exposing engine-native handles or secret data.

