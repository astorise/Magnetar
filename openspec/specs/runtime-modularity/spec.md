# runtime-modularity Specification

## Purpose
This specification defines how the Runtime's code is organised: the layers its modules belong to, the crates that hold those layers, the dependency direction between them and its enforcement, the stability of the public paths of `magnetar-runtime`, and the placement of tests next to the code they test.
## Requirements
### Requirement: Runtime Code Layers

Runtime code SHALL be organised in layers: core types, graph and memory, model data, kernel pipeline, Component engine, and Runtime orchestration. Every Runtime module SHALL be assigned to exactly one layer in a version-controlled layer map. A module SHALL depend only on modules of its own layer or of a lower layer.

#### Scenario: Lower layer does not reach the Inference API

- **WHEN** the dependencies of a core, graph, model data, kernel or Component engine module are listed
- **THEN** none of them is a Runtime orchestration module such as `inference_api`, `runtime` or `first_native_runtime`

#### Scenario: Module without a layer

- **WHEN** a Runtime module is added without an entry in the layer map
- **THEN** the layering check fails and names the module

### Requirement: Layer Crates

The core types, graph and memory, model data, kernel pipeline and Component engine layers SHALL each be built as a separate crate. The crate dependency graph SHALL be acyclic and SHALL follow the layer order. A module MAY stay in the Runtime orchestration crate when moving it lower would change a public type; the layer map SHALL record the reason.

#### Scenario: Core crate builds alone

- **WHEN** the core types crate is built on its own
- **THEN** it builds without the graph, kernel, Component engine or Runtime orchestration crates, and without a WebAssembly engine dependency

#### Scenario: Module kept in the orchestration crate

- **WHEN** a module assigned to a lower layer in the target layering stays in the Runtime orchestration crate
- **THEN** the layer map states which dependency prevents the move

### Requirement: Public Path Stability

The reorganisation SHALL NOT remove or rename a public item of `magnetar-runtime`. Every public path that resolved before the reorganisation SHALL resolve after it to the same item, including module paths and root re-exports.

#### Scenario: Consumer builds unchanged

- **WHEN** `magnetar-cli`, `inference-components`, `roadmap-contracts`, the integration test crates and the pinned Component, Provider, Format and Loader repositories are built against the reorganised Runtime
- **THEN** they compile and their tests pass without a source change

#### Scenario: Public API comparison

- **WHEN** the public API listing of `magnetar-runtime` is compared with the listing taken before the reorganisation
- **THEN** no item is missing and no signature differs

#### Scenario: Provider trait implementation

- **WHEN** an external Provider implements a Runtime trait for a type that moved to a lower-layer crate
- **THEN** the implementation still applies, because the type is defined once and re-exported

### Requirement: Behaviour Preservation

The reorganisation SHALL NOT change Runtime behaviour. Numerical results, conformance reports, error values and WIT contracts SHALL be identical before and after.

#### Scenario: Test suites

- **WHEN** the unit, contract, conformance and integration suites run after a reorganisation step
- **THEN** the same tests exist as before the step and all pass, with no test skipped, disabled or deleted

#### Scenario: WIT contracts

- **WHEN** the WIT packages are compared before and after the reorganisation
- **THEN** package names, interfaces, types and versions are identical

### Requirement: First-Native Module Responsibilities

First-native Runtime code SHALL be divided into modules with one responsibility each: model executor contract, fixture, KV state, multi-device segments, prepared plans, production generation, streaming, Component graph construction, chat session, and conformance report. No such module SHALL exceed 2,500 lines of non-test code. Tests SHALL sit with the module they test.

#### Scenario: Module size

- **WHEN** the non-test lines of each first-native module are counted
- **THEN** none exceeds 2,500

#### Scenario: Test visibility

- **WHEN** a test needs an item that was private in the former single file
- **THEN** the item is visible to the crate or the parent module only, and is not added to the public API

### Requirement: Roadmap Modules Are Leaves

No core, graph, kernel or Component engine module SHALL depend on the Provider roadmap or Model Format roadmap modules. Types that those layers need SHALL be defined in the module that owns them and MAY be re-exported from the roadmap modules.

#### Scenario: Kernel Artifact does not depend on a roadmap module

- **WHEN** the dependencies of the Kernel Artifact module are listed
- **THEN** the Provider roadmap module is not among them

#### Scenario: Existing roadmap paths

- **WHEN** a consumer names a type through its former roadmap module path
- **THEN** the path resolves to the same type

### Requirement: Sealed Authorities Stay Sealed

Moving a type to a lower-layer crate SHALL NOT make public a field or constructor that was crate-private in order to seal an authority. Code that needs the sealed item SHALL live in the crate that owns it, or SHALL use the public API.

#### Scenario: Model trust decision

- **WHEN** a crate other than the one defining the Model trust decision needs a decision
- **THEN** it obtains one from a Model trust store, and cannot construct one or write its fields

#### Scenario: Item widened for a higher crate

- **WHEN** a crate-private item becomes public because a higher-layer crate calls it
- **THEN** the item carries no authority, and it is hidden from the generated documentation

