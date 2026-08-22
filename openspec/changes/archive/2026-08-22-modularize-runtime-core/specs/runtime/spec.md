## ADDED Requirements

### Requirement: Modular Runtime Architecture

The `magnetar-runtime` implementation SHALL be organized into modules that
reflect canonical architectural responsibilities.

At minimum, implementation ownership SHALL distinguish:

- Runtime
- Component
- Capability
- Provider
- Device
- Resource Affinity
- Resolution
- Compute
- Planning
- Scheduler
- Observability

#### Scenario: Locate Provider implementation

Given a developer needs to modify Provider registration

When they inspect the Runtime source tree

Then Provider registration is owned by the Provider module

And is not implemented as unrelated code in the crate root.

---

### Requirement: Runtime Core Foundation

The modularized Runtime Core SHALL provide the implementation foundation for
future AI domains without creating those domains before their contracts exist.

The implementation SHALL distinguish the current Runtime Core from future AI
layers such as Model, Generation, and Agent.

#### Scenario: Prepare future AI domains

Given Runtime Core modularization is complete

When a future Model, Generation, or Agent change is proposed

Then it can consume the appropriate Runtime Core modules

And it does not need to add implementation directly to `lib.rs`.

---

### Requirement: Crate Root Is a FaÃ§ade

`magnetar-runtime/src/lib.rs` SHALL primarily act as the crate faÃ§ade.

It MAY contain:

- module declarations
- crate-level documentation
- intentional public re-exports
- minimal crate-wide constants when ownership is truly global

It SHALL NOT remain the main implementation file for unrelated Runtime
subsystems.

#### Scenario: Inspect crate root

Given the Runtime has been modularized

When a developer opens `lib.rs`

Then the primary architectural modules are visible

And detailed Provider, Compute, Scheduler, and Observability implementations are
located in their owning modules.

---

### Requirement: Module Boundaries Follow Responsibility

Source modules SHALL be organized by architectural responsibility rather than
arbitrary source-file size.

Splitting one coherent domain into files MAY be used for maintainability.

Unrelated responsibilities SHALL NOT be grouped merely because they share
helper code.

#### Scenario: Split Compute implementation

Given Compute contains tensor descriptors, graphs, schemas, movement, and
errors

When Compute is modularized

Then those files may remain within one Compute domain

Rather than being distributed across unrelated generic utility modules.

---

### Requirement: Runtime Owns Orchestration

The Runtime module SHALL own top-level local orchestration.

Runtime SHALL coordinate subsystem behavior without absorbing implementation
ownership from those subsystems.

#### Scenario: Initialize Runtime

Given a Runtime is created

When initialization occurs

Then the Runtime coordinates required subsystem initialization

While Provider, Component, Capability, and other domain logic remains owned by
their respective modules.

---

### Requirement: Component Module Ownership

Component-related portable extension concepts SHALL be owned by the Component
module.

This includes current Component metadata, lifecycle abstraction, management,
and Component-specific errors.

#### Scenario: Inspect Component state

Given Component lifecycle state is required

When its implementation is located

Then it belongs to the Component domain

And not to Provider or Runtime configuration code.

---

### Requirement: Capability Module Ownership

Capability identity, versioning, description, compatibility, and registration
SHALL be owned by the Capability domain.

Capability definitions SHALL remain independent from concrete Provider
implementation.

#### Scenario: Check Capability version

Given the Runtime validates Capability compatibility

When semantic-version behavior is applied

Then that behavior is owned by the Capability domain

And does not require Provider-specific logic.

---

### Requirement: Provider Module Ownership

Trusted native execution extension behavior SHALL be owned by the Provider
domain.

Provider responsibilities MAY include:

- Provider metadata
- Provider registry
- Provider discovery/loading
- Capability advertisement
- Compute advertisement
- native execution API
- health integration
- Provider-specific errors

#### Scenario: Load native Provider

Given a native Provider library is discovered

When it is loaded and registered

Then Provider-owned code handles the native extension semantics

And no Component or generic Plugin module owns the operation.

---

### Requirement: Device Module Ownership

Device identity and Device metadata SHALL be owned by the Device domain.

Device representation SHALL remain usable by Provider and Runtime resolution
without depending on higher-level application semantics.

#### Scenario: Describe GPU

Given a Provider exposes a GPU Device

When its metadata is represented

Then Device-owned types describe the execution target

And those types do not depend on future Model or Agent implementations.

---

### Requirement: Resource Affinity Module Ownership

Resource binding and affinity semantics SHALL be owned by a dedicated Resource
Affinity domain.

This domain MAY own:

- ProviderBinding
- DeviceBinding
- CapabilityBinding
- ArtifactBinding
- AffinityGroupId
- affinity constraint structures
- affinity validation

#### Scenario: Validate bound resource

Given a tensor is bound to one Provider

When dependent execution is validated

Then affinity-specific validation is provided by the Resource Affinity domain.

---

### Requirement: Resolution Module Ownership

Provider and Device candidate resolution SHALL be owned by a Resolution domain.

Resolution SHALL evaluate compatible candidates using established constraints
and policy.

Resolution SHALL NOT own Scheduler queue execution.

#### Scenario: Resolve Provider

Given multiple Providers implement a Capability

When target selection occurs

Then Resolution evaluates candidates and policy

And the Scheduler later consumes the resolved plan.

---

### Requirement: Compute Module Ownership

Portable Compute domain models SHALL be owned by the Compute module.

The Compute module SHALL remain independent from concrete hardware execution
implementations.

#### Scenario: Define tensor descriptor

Given a Tensor Descriptor is part of the portable Compute contract

When its Rust domain model is located

Then it belongs to Compute

And does not require CUDA, CPU, or another Provider implementation.

---

### Requirement: Planning Module Ownership

Memory and execution planning SHALL be represented as planning responsibilities
separate from scheduling.

Planning SHALL determine how validated work can execute.

Scheduling SHALL determine when validated planned work executes.

#### Scenario: Prepare graph execution

Given a validated Compute graph

When execution preparation occurs

Then Planning produces the required execution and memory decisions

Before Scheduler admission.

---

### Requirement: Scheduler Module Ownership

Scheduler SHALL own operation admission, queueing, submission timing,
cancellation coordination, and operation lifecycle scheduling.

Scheduler SHALL consume validated execution plans.

Scheduler SHALL NOT independently perform global Provider resolution.

#### Scenario: Schedule resolved execution

Given an execution plan already selects its Provider and Device

When the Scheduler accepts the operation

Then it schedules that plan

And does not independently select another Provider.

---

### Requirement: Observability Module Ownership

Runtime observability implementation SHALL be owned by the Observability domain.

Observability SHALL remain separated from execution correctness.

#### Scenario: Exporter becomes slow

Given an observability exporter cannot consume observations quickly enough

When Compute execution proceeds

Then the Observability module applies its bounded delivery policy

And Compute correctness is unaffected.

---

## ADDED Requirements

### Requirement: Architectural Dependency Direction

Module dependencies SHALL preserve architectural layering.

Lower-level native execution abstractions SHALL NOT depend on future
application-level AI features.

Prohibited conceptual dependencies include:

- Provider depending on Agent
- Provider depending on Generation
- Device depending on Model
- Compute depending on Tachyon
- Affinity depending on Scheduler queue state

#### Scenario: Add future Agent domain

Given a future Agent module is introduced

When dependencies are defined

Then Agent may consume Runtime services

But Provider does not begin depending on Agent.

---

### Requirement: No Tachyon Runtime Dependency

`magnetar-runtime` modules SHALL NOT depend on Tachyon implementation packages.

Tachyon integration SHALL consume Magnetar through explicit external
integration contracts.

#### Scenario: Modularization completes

Given all Runtime modules have been reorganized

When dependencies are reviewed

Then none requires a Tachyon crate or Tachyon-specific Runtime type.

---

### Requirement: Avoid Generic Common Module

Magnetar SHALL NOT create a generic common or utility module as a dumping
ground for unrelated architectural types.

Cross-domain primitives SHALL have explicit ownership.

#### Scenario: Shared Provider identity

Given multiple modules need ProviderBinding

When its source ownership is chosen

Then it belongs to Resource Affinity or another explicit architectural domain

And not to an undefined generic `common` module.

---

## ADDED Requirements

### Requirement: Explicit Public API

Public visibility SHALL be intentional.

A type SHALL NOT remain public merely because it previously lived in
`lib.rs`.

#### Scenario: Internal planning helper

Given a helper is only used while constructing execution plans

When code is modularized

Then it becomes private or `pub(crate)` unless external consumers require it.

---

### Requirement: Intentional Crate-Root Re-Exports

The crate root MAY re-export canonical public APIs for ergonomics.

Public re-exports SHALL be explicit.

Wildcard re-export of entire implementation modules SHOULD be avoided.

#### Scenario: Re-export Runtime

Given `Runtime` remains a primary public entry point

When the source moves to `runtime`

Then the crate root may explicitly re-export `Runtime`

Without re-exporting every Runtime implementation helper.

---

### Requirement: Preserve Canonical API Semantics

Structural module movement SHALL NOT silently change canonical public
semantics.

Intentional semantic changes SHALL require their own architecture change unless
they are necessary to complete an already approved preceding change.

#### Scenario: Move Resolution Policy

Given Resolution Policy is moved to a new source module

When existing tests execute

Then candidate ordering and affinity precedence remain unchanged.

---

## ADDED Requirements

### Requirement: WIT Representation Isolated from Internal Domain Models

Portable WIT representations SHOULD be isolated from internal Runtime
representations where this separation prevents coupling.

WIT-generated types SHALL NOT automatically become the canonical internal
representation of all Runtime state.

#### Scenario: Receive Compute descriptor

Given a Component submits a WIT Compute descriptor

When the Runtime accepts it

Then the Runtime may validate and convert the descriptor into an internal
Compute domain representation.

---

### Requirement: Native Types Do Not Leak Through WIT Modules

Provider-native types SHALL NOT become dependencies of Component-facing WIT
models.

#### Scenario: Convert Compute request

Given a Compute WIT request is converted to internal state

When concrete Provider placement is resolved

Then ProviderBinding and DeviceBinding remain Runtime-native state

And are not added to the portable request model.

---

## ADDED Requirements

### Requirement: Unit Tests Follow Domain Ownership

Domain-specific unit tests SHOULD reside with the module owning the behavior.

#### Scenario: Test Capability compatibility

Given semantic Capability version compatibility is tested

When tests are reorganized

Then the tests reside with Capability implementation or another clearly
Capability-owned test location.

---

### Requirement: Cross-Domain Behavior Uses Integration Tests

Behavior spanning several architectural modules SHOULD be validated through
integration tests.

Important integration paths SHALL include:

- Capability to Provider resolution
- Resource Affinity during resolution
- Compute to Planning
- Planning to Scheduler
- Scheduler to Provider execution
- Runtime to Observability

#### Scenario: Test complete execution path

Given a synthetic Provider and portable Compute request

When an integration test executes the Runtime path

Then resolution, planning, scheduling, and Provider execution can be validated
without depending solely on private module tests.

---

### Requirement: Structural Refactor Preserves Test Coverage

Production code moved into new modules SHALL remain inside the repository's
protected coverage scope.

Modularization SHALL NOT reduce measured coverage merely because files changed
location.

#### Scenario: Move Provider code

Given Provider implementation moves out of `lib.rs`

When CI coverage runs

Then the Provider module remains part of production coverage measurement.

---

## ADDED Requirements

### Requirement: No Premature Workspace Fragmentation

This change SHALL modularize the existing Runtime crate before splitting the
architecture into multiple workspace crates.

#### Scenario: Provider module is created

Given Provider has a clear source module

When this change completes

Then a separate `magnetar-provider` crate is not required solely because a
Provider module exists.

---

### Requirement: Future Crate Extraction Requires Evidence

A future module-to-crate extraction SHOULD be motivated by demonstrated needs
such as:

- independent dependency boundaries
- independent compilation
- reuse
- ABI or API isolation
- optional features
- independent testing
- build performance
- versioning

#### Scenario: Consider extracting Compute

Given the Compute module has stabilized

When a future crate extraction is proposed

Then the proposal explains the concrete benefit of a separate crate.

---

## ADDED Requirements

### Requirement: Do Not Create Empty Future Architecture

This modularization SHALL NOT introduce empty production modules solely for
future roadmap domains.

The following domains SHALL be introduced through their dedicated changes:

- Model
- Generation
- Inference
- KV cache
- Adapter
- Agent
- Tool

#### Scenario: Modularization completes before Model work

Given Model execution has not yet been standardized

When the Runtime tree is reorganized

Then no empty `model` production module is required merely as a placeholder.

---

### Requirement: Future AI Domains Build Above the Modular Core

Future Model, Generation, Inference, Agent, and Tool functionality SHALL build
upon the modular Runtime rather than being inserted back into the crate root.

#### Scenario: Add Generation later

Given a future Generation change is implemented

When its source module is added

Then it consumes appropriate Runtime, Capability, Compute, Planning, and
Provider abstractions

Without adding Generation implementation directly to `lib.rs`.

---

### Requirement: Future Component Engine Boundary Is Separate

This modularization SHALL NOT define or implement the concrete WASM Component
engine boundary.

The follow-up change `stabilize-component-runtime-boundary` SHALL define the
abstract Component Runtime boundary before a concrete engine such as Wasmtime is
selected or wired.

That later boundary MAY include:

- `ComponentEngine`
- compilation and validation
- `ComponentInstance`
- `Store`
- Capability linking
- resources
- traps
- cancellation
- sandboxing
- separation between the abstract Runtime Component contract and the Wasmtime implementation

#### Scenario: Defer Wasmtime integration

Given the Runtime Core has been modularized

When Component engine work begins

Then `stabilize-component-runtime-boundary` defines the engine-facing contracts

And this modularization has not already introduced a Wasmtime-specific
implementation.

---

## ADDED Requirements

### Requirement: No Duplicate Implementations During Completion

At completion of the modularization, one canonical implementation SHALL exist
for each migrated architectural concept.

Temporary forwarding code MAY exist during implementation but SHALL NOT leave
parallel implementations behind.

#### Scenario: Provider registry moved

Given ProviderRegistry is migrated to `provider`

When the change completes

Then no second ProviderRegistry implementation remains in `lib.rs`.

---

### Requirement: No Semantic Changes Hidden as Refactoring

Unexpected architectural issues discovered during modularization SHALL NOT be
silently redesigned as part of source movement.

They SHALL be documented and, when materially semantic, addressed through a
dedicated OpenSpec change.

#### Scenario: Circular dependency exposes model issue

Given modularization reveals that two domains depend on each other because an
existing responsibility is misplaced

When resolving the issue would change public Runtime semantics

Then the semantic redesign is proposed separately

Rather than hidden inside file movement.

---

### Requirement: Crate Root Size Is Not the Goal

The objective of this change SHALL be architectural modularity rather than an
arbitrary maximum line or byte count.

Large cohesive modules MAY remain large when their responsibility is clear.

#### Scenario: Compute module remains substantial

Given the Compute domain contains many coherent schemas and descriptors

When modularization completes

Then the module may remain sizeable

Provided its ownership is clear and unrelated Runtime responsibilities are not
mixed into it.


