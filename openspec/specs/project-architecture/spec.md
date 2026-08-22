# project-architecture Specification

## Purpose
TBD - created by archiving change align-project-architecture-and-openspec-context. Update Purpose after archive.
## Requirements
### Requirement: Canonical Magnetar Architecture

Magnetar SHALL use Runtime, Component, Capability, Provider, Device, Resource
Affinity, and Resolution Policy as its canonical foundational architecture.

The canonical execution relationship SHALL be:

```text
Component
    |
    | imports Capability
    v
Runtime
    |
    | resolves according to policy and affinity
    v
Provider
    |
    v
Device
```

#### Scenario: Describe Magnetar execution

Given project documentation describes execution of a portable workload

When the execution path is represented

Then the Component requests a Capability

And the Runtime resolves a compatible Provider and Device.

---

### Requirement: Runtime Responsibility

The Runtime SHALL own local orchestration of Magnetar execution.

Runtime responsibilities MAY include:

- Component management
- Capability resolution
- Provider management
- Device discovery coordination
- Resource Affinity
- Resolution Policy
- memory planning
- execution planning
- scheduling
- Provider execution coordination
- observability
- recovery policy
- future model and inference services

#### Scenario: Resolve execution

Given a Component requests a Capability

When multiple compatible Providers are available

Then Runtime Resolution Policy determines the selected execution target.

---

### Requirement: Component Definition

A Component SHALL represent portable code executed through the WebAssembly
Component Model.

Components SHALL interact with Magnetar through portable contracts.

Components SHALL NOT be treated as trusted native Providers.

#### Scenario: Load model-related Component

Given a future model architecture Component

When Magnetar loads the Component

Then the Component interacts with Runtime services through authorized portable
interfaces

And does not receive direct native Device APIs.

---

### Requirement: Capability Definition

A Capability SHALL represent a portable WIT contract describing an ability
available to Components.

A Capability SHALL remain independent from a particular hardware
implementation.

#### Scenario: Same Capability on different hardware

Given CPU and CUDA Providers implement the same compatible Capability

When a Component imports that Capability

Then the Component contract does not change because the Runtime selected CPU or
CUDA.

---

### Requirement: Provider Definition

A Provider SHALL represent a native Runtime extension implementing one or more
Capabilities.

A Provider MAY own native implementation details including:

- kernels
- native libraries
- native execution contexts
- allocators
- queues
- streams
- driver APIs
- Device-native resources

These implementation details SHALL remain private to the Runtime-to-Provider
boundary.

#### Scenario: CUDA execution

Given a CUDA Provider implements a Compute Capability

When the Runtime submits Provider-bound work

Then CUDA-native implementation objects remain inside the native execution
boundary.

---

### Requirement: Device Definition

A Device SHALL represent a physical or logical execution target exposed by a
Provider.

Devices SHALL NOT be directly selected by portable Components.

#### Scenario: Select GPU

Given several GPU Devices are exposed by a Provider

When a Component requests portable compute

Then Runtime resolution selects an eligible Device according to policy,
advertisement, health, pressure, and Resource Affinity.

---

### Requirement: Runtime Owns Provider and Device Resolution

Portable Components SHALL NOT directly choose Providers or Devices.

Provider and Device selection SHALL remain a Runtime responsibility.

#### Scenario: Model Component executes compute

Given a model Component requests compute

When execution planning begins

Then the model Component does not choose `cuda`, `gpu:0`, or another native
execution target

And the Runtime resolves an eligible target.

---

### Requirement: Resource Affinity Is Authoritative

Resource Affinity SHALL constrain Runtime resolution whenever a resource is
bound to execution identity or state.

Affinity MAY include bindings to:

- Provider
- Device
- Capability implementation
- execution context
- artifact
- model
- adapter
- tokenizer
- prompt template
- cache
- future resource identities

#### Scenario: Provider-pinned live state

Given live state is bound to one Provider

When another Provider becomes otherwise preferable

Then Runtime resolution preserves the binding unless an explicit migration
mechanism exists.

---

### Requirement: Resolution Policy

Resolution Policy SHALL determine selection among compatible execution
candidates after mandatory compatibility and affinity constraints are applied.

Policy SHALL NOT override authoritative Resource Affinity.

#### Scenario: Preferred Provider conflicts with affinity

Given policy prefers Provider B

And a live resource is bound to Provider A

When execution is resolved

Then Provider A remains required while the binding is authoritative.

---

### Requirement: Backend Is Not a Canonical Architectural Concept

`Backend` SHALL NOT be used as a primary Magnetar architectural concept in new
canonical specifications or documentation.

Native hardware implementations SHALL be modeled as Providers.

Historical artifacts MAY retain the term when preservation of project history
requires it.

#### Scenario: Describe CUDA implementation

Given new Magnetar architecture documentation

When CUDA support is described

Then it is described as a CUDA Provider rather than a CUDA Backend.

---

### Requirement: Plugin Is Not a Canonical Architectural Concept

`Plugin` SHALL NOT be used as the primary term for new Magnetar extensions.

A native implementation SHALL use Provider terminology.

A portable WebAssembly extension SHALL use Component terminology.

#### Scenario: Add external integration

Given a new portable observability integration

When it is modeled

Then it is a Component rather than a Plugin or Provider.

---

### Requirement: Host Is Not a Primary Architectural Entity

`Host` SHALL NOT become a primary Magnetar architectural abstraction when
Runtime, Component, Provider, or Device more precisely describes the role.

The term MAY be used descriptively for platform or environment concepts where no
architectural identity is implied.

#### Scenario: Describe Component execution environment

Given documentation discusses Component execution

When architectural ownership is described

Then the Runtime remains the owning architectural concept rather than creating a
parallel Host abstraction.

---

**Component and Provider Boundary**

### Requirement: Components and Providers Are Distinct Extension Mechanisms

Magnetar SHALL maintain a strict distinction between portable Components and
native Providers.

Components SHALL be portable and sandboxable.

Providers SHALL be native and trusted.

#### Scenario: Compare OTEL and CUDA

Given an OpenTelemetry exporter and a CUDA compute implementation

When each integration is classified

Then the OpenTelemetry integration may be a WASM Component

And CUDA is implemented by a native Provider.

---

### Requirement: Component Native Isolation

Components SHALL NOT receive native implementation objects.

Forbidden portable values include:

- raw pointers
- GPU pointers
- backend storage objects
- Provider handles
- Device handles
- native queue handles
- native stream handles
- allocator objects
- kernel objects
- Rust trait objects
- process-local object references

#### Scenario: Return tensor resource

Given a Component receives a tensor resource

When it uses the resource

Then storage ownership remains opaque

And native implementation state remains inside the Runtime and Provider
boundary.

---

### Requirement: Coarse Component Execution

Portable compute contracts SHALL favor coarse-grained graph, batch, model,
session, or equivalent execution units.

Magnetar SHALL NOT require a WIT transition for every eager tensor primitive.

#### Scenario: Model executes transformer work

Given a Component needs multiple tensor operations

When it submits the work

Then operations may be represented as a graph or equivalent coarse execution
unit rather than one cross-WASM call per primitive.

---

**AI Runtime Scope**

### Requirement: Magnetar Owns AI Execution

Magnetar SHALL be the architectural owner of local AI execution.

Future Magnetar responsibilities MAY include:

- model loading
- model residency
- tokenization
- prompt formatting
- generation
- streaming
- continuous batching
- KV cache management
- prefix caching
- adapters
- quantization
- speculative decoding
- structured generation
- multi-device execution
- agent execution
- tool orchestration

These future responsibilities SHALL NOT be documented as already implemented
until their respective changes are completed.

#### Scenario: Introduce continuous batching

Given continuous batching is implemented for AI inference

When architectural ownership is determined

Then the local inference batching implementation belongs to Magnetar.

---

### Requirement: Model Architecture Is Not a Provider

Model architecture SHALL remain conceptually separate from hardware execution
Provider implementation.

Model families such as Llama, Qwen, Gemma, or DeepSeek SHALL NOT be modeled as
Providers solely because they define model architecture.

#### Scenario: Add Qwen support

Given Qwen model architecture support is introduced

When the architecture is modeled

Then it is represented through model, Runtime, or Component abstractions

And not through a `QwenProvider` whose purpose is merely model definition.

---

### Requirement: Model Components May Be Portable

Magnetar SHALL allow model architecture logic to be implemented as WASM
Components where the boundary is portable and performance requirements permit
it.

Provider-owned low-level kernels SHALL remain native.

#### Scenario: Portable Qwen architecture

Given Qwen architecture logic can construct or request coarse compute work using
portable Magnetar contracts

When a Qwen Component is used

Then hardware-specific execution remains delegated to a selected Provider.

---

**Magnetar and Tachyon**

### Requirement: Magnetar and Tachyon Are Independent Layers

Magnetar SHALL NOT require Tachyon for standalone Runtime operation.

Tachyon MAY consume and extend Magnetar.

The canonical dependency direction SHALL permit:

```text
Tachyon
   |
   v
Magnetar
```

without requiring:

```text
Magnetar
   |
   v
Tachyon
```

#### Scenario: Run Magnetar locally

Given Tachyon is not installed

When a user starts Magnetar locally

Then Magnetar can operate without requiring Tachyon services.

---

### Requirement: Tachyon Owns Distributed Service Orchestration

Tachyon SHALL own distributed responsibilities outside local Magnetar execution when those responsibilities are present, including:

- cluster membership
- service mesh behavior
- inter-node routing
- distributed deployment
- GitOps
- node selection
- cluster-level availability
- distribution of Components and artifacts

These responsibilities SHALL remain outside Magnetar local execution semantics.

#### Scenario: Route inference to a node

Given several Tachyon nodes can execute a model

When one cluster node must be selected

Then Tachyon may select the node

And Magnetar performs local execution after the request reaches that node.

---

### Requirement: Magnetar Owns Intra-Node Inference Scheduling

Future inference batching and scheduling inside one Magnetar Runtime SHALL belong
to Magnetar.

Tachyon SHALL NOT require a duplicate model-specific intra-node inference
scheduler after inference migration is complete.

#### Scenario: Continuous batching

Given multiple inference requests arrive on one node

When prefill and decode work are batched

Then Magnetar owns the local batching and Device scheduling mechanics.

---

### Requirement: Tachyon May Distribute Magnetar Components

Tachyon SHALL remain an optional external distributor when it provides WASM Components compatible with Magnetar.

Magnetar SHALL remain responsible for:

- Component validation
- compatibility validation
- Capability linking
- authority enforcement
- sandbox execution

#### Scenario: Tachyon distributes coding agent

Given Tachyon distributes a Magnetar-compatible coding-agent Component to a node

When Magnetar receives the artifact

Then Magnetar validates and executes the Component according to local Runtime
policy.

---

### Requirement: Component Distribution Is Vendor-Neutral

The Magnetar Component distribution boundary SHALL NOT require Tachyon-specific
semantics.

Tachyon SHALL be one possible Component source rather than a mandatory source.

#### Scenario: Local Component installation

Given a user installs a Component without Tachyon

When Magnetar validates the same artifact format

Then the Component can be used locally.

---

**Artifact Model Terminology**

### Requirement: Component Artifact and Model Artifact Are Distinct

Magnetar SHALL distinguish executable Component artifacts from model artifacts.

A Component Artifact represents executable WASM Component code.

A Model Artifact represents model data such as:

- weights
- configuration
- tokenizer data
- model metadata

#### Scenario: Load Qwen

Given a Qwen implementation uses both portable architecture code and model
weights

When Magnetar loads them

Then the executable Component Artifact and Model Artifact remain separately
identified resources.

---

### Requirement: Future Model Instance Composition

A future loaded model instance SHALL keep these identities distinguishable when it combines:

- model architecture implementation
- Model Artifact
- optional Component Artifact
- Provider
- Device or Device group
- execution resources
- Resource Affinity

This requirement SHALL NOT prescribe the final model lifecycle contract before
its dedicated OpenSpec change.

#### Scenario: Resident model

Given a future model becomes resident on a GPU

When its execution state is represented

Then its model artifact identity and native execution affinity can remain
distinguishable.

---

**magnetar-cli**

### Requirement: magnetar-cli Is a Runtime Client

`magnetar-cli` SHALL be designed as a first-party client of Magnetar Runtime
services rather than as an independent inference implementation.

#### Scenario: Local generation

Given a user runs a future command such as:

`magnetar run <model> <prompt>`

When generation executes

Then the CLI invokes Magnetar Runtime functionality

And does not maintain a separate model execution engine.

---

### Requirement: Coding Agent Logic Is Not CLI Infrastructure

Future Codex-like coding functionality SHALL be built from Magnetar agent,
conversation, generation, and tool capabilities rather than being embedded as
unrelated inference logic inside the CLI frontend.

#### Scenario: Start coding agent

Given a user starts a future coding-agent mode

When the agent operates

Then `magnetar-cli` provides the user interface

And Magnetar Runtime services own execution and authorized tool access.

---

**Runtime Consumption Modes**

### Requirement: Shared Runtime Semantics

Future embedded, CLI, service, and Tachyon-integrated Magnetar usage SHALL share
the same core Runtime semantics.

A frontend SHALL NOT redefine Provider selection, Resource Affinity, scheduling,
or model execution semantics.

#### Scenario: Same model via CLI and service

Given the same execution policy and resources

When a request is submitted through CLI or a future service API

Then both paths use the same Magnetar Runtime architecture.

---

**OpenSpec Governance**

### Requirement: OpenSpec Project Context

`openspec/config.yaml` SHALL contain Magnetar-specific project context.

The context SHALL describe at least:

- canonical terminology
- architecture
- Component/Capability/Provider/Device relationship
- Resource Affinity
- Resolution Policy
- Component versus Provider distinction
- native isolation rules
- Magnetar/Tachyon responsibility boundary
- future AI Runtime scope
- standalone Magnetar requirement

#### Scenario: Generate future OpenSpec change

Given an OpenSpec authoring agent reads project context

When it prepares a new Magnetar change

Then the canonical architecture is available without requiring conversation
history.

---

### Requirement: OpenSpec Architectural Ownership

A change introducing new Runtime behavior SHALL identify the architectural layer
that owns the behavior.

#### Scenario: Propose model cache

Given a change introduces a model cache

When the proposal is reviewed

Then it explains whether the cache belongs to Runtime, Component, Provider,
Device, or another established layer.

---

### Requirement: OpenSpec Capability Requirements

A proposal introducing or changing portable Component behavior SHALL identify
the relevant Capability/WIT boundary.

#### Scenario: Add tool Component

Given a change proposes a tool Component

When its contract is defined

Then its imported and exported portable interfaces are explicit.

---

### Requirement: OpenSpec Provider Requirements

A proposal introducing a Provider SHALL identify:

- implemented Capabilities
- exposed Device types
- native responsibilities
- compatibility expectations
- Resource Affinity behavior where applicable

#### Scenario: Add CUDA Provider

Given a CUDA Provider change

When the proposal is reviewed

Then the Provider's native execution responsibilities and portable advertised
Capabilities are explicit.

---

### Requirement: OpenSpec Affinity Requirements

Changes introducing stateful or opaque resources SHALL describe Resource
Affinity semantics.

#### Scenario: Add KV cache resource

Given a future KV cache resource

When its change is proposed

Then the proposal states whether the cache is Provider-bound, Device-bound,
restartable, or otherwise constrained.

---

### Requirement: OpenSpec Recovery Requirements

Changes introducing execution or stateful operations SHALL document recovery
semantics when failure can occur.

#### Scenario: Stateful generation failure

Given a future generation session can fail after state creation

When its change is proposed

Then retry, replay, Provider pinning, or abort semantics are explicitly
addressed.

---

### Requirement: OpenSpec Authority Requirements

Changes introducing Component access to external resources SHALL document
authority and scope requirements.

External resources MAY include:

- network access
- filesystem access
- secrets
- process execution
- source-control access
- external services

#### Scenario: Coding tool accesses filesystem

Given a future tool Component needs filesystem access

When its change is proposed

Then filesystem authority and scope are explicit.

---

### Requirement: Canonical Documentation

Magnetar SHALL maintain a canonical architecture overview linked from the
repository README.

Detailed architecture documents MAY specialize individual areas, but SHALL NOT
contradict the canonical architecture without an explicit architecture change.

#### Scenario: Reader discovers architecture

Given a developer opens the repository README

When they follow the architecture documentation

Then they can determine the canonical roles of Runtime, Components,
Capabilities, Providers, Devices, Magnetar, and Tachyon.

---

### Requirement: Implemented and Planned Features Are Distinct

Documentation SHALL distinguish functionality already implemented from
architectural roadmap functionality.

#### Scenario: README mentions agent Runtime

Given agent execution has not yet been implemented

When the README describes the project direction

Then agent execution is clearly identified as planned rather than existing
functionality.

---

### Requirement: Historical OpenSpec Artifacts Remain Historical

Archived OpenSpec changes SHALL preserve historical context.

They SHALL NOT be mass-rewritten merely to replace deprecated terminology.

Current canonical specifications and architecture documentation SHALL define the
present architecture.

#### Scenario: Archived add-plugin-system change exists

Given the historical `add-plugin-system` change remains in the archive

When current architecture is interpreted

Then the canonical Provider/Component terminology takes precedence over that
historical terminology.

