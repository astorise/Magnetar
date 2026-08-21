## Purpose

Define the runtime capability model and its requirements.
## Requirements
### Requirement: Capability Identity

Every Capability SHALL expose a globally unique identifier.

#### Scenario: Register capability

Given a Capability

When it is registered

Then its identifier SHALL be unique.

---

### Requirement: Capability Versioning

Capabilities SHALL follow semantic versioning.

#### Scenario: Multiple versions

Given several versions of the same Capability

When compatibility is evaluated

Then semantic versioning rules SHALL apply.

---

### Requirement: Capability Contracts

Every Capability SHALL expose one or more WIT contracts.

#### Scenario: Import capability

Given a Component importing a Capability

When the Runtime validates dependencies

Then every required WIT interface SHALL be present.

---

### Requirement: Capability Dependencies

Capabilities SHALL declare every Capability dependency required for execution.

#### Scenario: Resolve dependencies

Given a Capability requiring another Capability

When the Runtime initializes

Then dependencies SHALL be resolved before execution.

---

### Requirement: Capability Resolution

The Runtime SHALL resolve Capabilities independently from Providers.

#### Scenario: Resolve execution

Given multiple Providers implementing the same Capability

When execution begins

Then the Runtime SHALL select a compatible Provider.

---

### Requirement: Capability Independence

Capabilities SHALL remain independent from hardware implementations.

#### Scenario: Multiple hardware targets

Given identical Capability contracts

When different Providers implement them

Then Components SHALL execute without modification.

### Requirement: Compute Capability

The Runtime SHALL expose a Compute Capability.

Components SHALL use this capability for mathematical execution.

The Compute Capability identifier SHALL be `magnetar:compute/run` at version
`1.1.0`.

#### Scenario: Resolve the canonical Compute capability

Given a Provider advertising the Compute Capability

When a Component imports `magnetar:compute/run@1.1.0`

Then the Runtime SHALL resolve that Provider without exposing it to the
Component.

---

### Requirement: Hardware Independence

The Compute Capability SHALL remain independent from hardware
implementations.

#### Scenario: Execute on different Providers

Given multiple Providers implementing Compute

When a Component executes

Then no Component modification is required.

---

### Requirement: WIT Contract

The Compute Capability SHALL be defined through WIT.

Its initial WIT package SHALL be `magnetar:compute@1.0.0` and expose the
`run` interface as a marker contract with no concrete mathematical operations.

The executable Compute WIT package SHALL be `magnetar:compute@1.1.0` and expose
the `run` interface with coarse graph submission, opaque tensor and graph
resources, operation lifecycle functions, portable tensor descriptors, and
stable structured errors.

#### Scenario: Validate interface

Given a Compute Provider advertising `magnetar:compute/run@1.1.0`

When registration occurs

Then the Provider SHALL implement the `magnetar:compute@1.1.0` `run` WIT
contract.

---

### Requirement: Version Compatibility

Compute implementations SHALL support semantic versioning.

For stable major versions, an available Compute capability SHALL satisfy an
equal-or-earlier requirement with the same major version. Breaking WIT changes
SHALL use a new major version.

#### Scenario: Capability upgrade

Given multiple versions of the Compute Capability

When compatibility is evaluated

Then semantic versioning rules SHALL apply.

### Requirement: Evidence-Based Capability Derivation

Every Capability family proposed by a source-derived taxonomy SHALL identify
the pinned external runtime revision, relevant source path, and source symbol
that motivate its boundary.

#### Scenario: Trace a proposed family

- **WHEN** a reviewer inspects a Capability family proposed by the Candle and
  Crane source-derived taxonomy
- **THEN** the reviewer can trace it to at least one Candle or Crane source
  symbol at a recorded revision

### Requirement: Layered Responsibility Taxonomy

The contract-derivation taxonomy SHALL classify source responsibilities into
low-level execution, model-level execution, or application-level abilities and
SHALL map each responsibility to Magnetar's Provider, Capability, Component,
Device, or runtime-service roles.

#### Scenario: Classify a runtime responsibility

- **WHEN** an existing runtime responsibility is mapped into Magnetar
- **THEN** it has one primary taxonomy layer, a target Magnetar role, stated
  responsibilities, and explicit exclusions

### Requirement: Capability Candidate Qualification

A responsibility SHALL be proposed as a Capability family only when it has a
portable WIT boundary; native-only responsibilities SHALL remain Provider,
Device, or runtime implementation details.

#### Scenario: Classify a native backend surface

- **WHEN** a source responsibility depends on native resource representation
  and has no portable contract boundary
- **THEN** it is excluded from the Capability candidates and assigned to the
  appropriate native Magnetar role

### Requirement: Capability Dependency Documentation

Every proposed Capability family SHALL document the other Capability families
or existing Magnetar runtime services required for its execution.

#### Scenario: Review a candidate contract

- **WHEN** a candidate contract is prepared for a future WIT change
- **THEN** its transitive execution dependencies are visible from the taxonomy

### Requirement: Native and Component Boundary Classification

Every analyzed responsibility SHALL be classified as native-only,
Component-suitable, or hybrid. Hybrid candidates SHALL distinguish their WIT
surface from Provider-owned resources, with rationale covering ownership, data
transfer, state, and call granularity.

#### Scenario: Select a contract boundary

- **WHEN** a responsibility owns device resources or crosses a portable
  Component boundary
- **THEN** the taxonomy explains which part remains native and which part can be
  expressed through WIT

### Requirement: Fallback Semantics Classification

Every proposed Capability family SHALL document whether fallback is
transparent, restartable from replayable input, or pinned for the lifetime of
Provider-owned state.

#### Scenario: Provider failure during execution

- **WHEN** a Provider fails before or during a Capability operation
- **THEN** the taxonomy states whether another compatible Provider can be used
  without violating resource ownership or observable stream semantics

### Requirement: Provisional WIT Package Mapping

Component-suitable and hybrid Capability families SHALL be mapped to candidate
WIT packages without registering those candidates as stable runtime contracts.

#### Scenario: Prepare a follow-up WIT proposal

- **WHEN** a future change selects a Capability family for standardization
- **THEN** it has a provisional package boundary and the unresolved contract
  decisions needed before versioning

### Requirement: Magnetar-Native Contract Shape

Derived contracts SHALL express Magnetar's Provider, Capability, Component,
and Device model rather than copying Candle or Crane APIs.

#### Scenario: Adapt an existing interface

- **WHEN** a Candle or Crane interface motivates a Magnetar boundary
- **THEN** Rust-specific generics, callbacks, storage implementations, and
  concrete model types are excluded from the portable contract shape

### Requirement: Coarse Compute Submission

`magnetar:compute/run` SHALL represent coarse compute submission.

Components SHALL NOT call one WIT function per eager tensor primitive.

#### Scenario: Submit compute work

- **GIVEN** a Component with compute work represented by an opaque graph
- **WHEN** it calls `magnetar:compute/run`
- **THEN** the work is submitted as a graph, batch, or equivalent coarse unit

---

### Requirement: Portable Tensor Descriptors

Tensor metadata SHALL be represented with portable fixed-width descriptors.

Descriptors SHALL include shape, dtype, and view metadata.

Descriptors SHALL NOT expose Rust `usize`, platform-dependent layout objects,
or backend-specific storage enums.

#### Scenario: Validate descriptor

- **GIVEN** a tensor descriptor
- **WHEN** the Runtime validates it before execution
- **THEN** shape, dtype, view, and size constraints are checked without
  depending on platform word size

---

### Requirement: Opaque Provider-Owned Resources

Tensor storage and graph representation SHALL remain opaque to Components.

Components SHALL NOT receive raw pointers, backend storage, GPU handles, native
tensor objects, Provider handles, or Device handles through the Compute WIT
contract.

#### Scenario: Use tensor and graph resources

- **GIVEN** a Component receives opaque tensor and graph resources
- **WHEN** it submits compute execution
- **THEN** the Runtime preserves Provider-owned storage, graph representation,
  affinity, and lifetime internally

---

### Requirement: Compute Operation Lifecycle

Compute execution SHALL expose an operation resource with explicit status,
await, cancellation, and output retrieval semantics.

An operation SHALL eventually reach a terminal state of completed, cancelled,
or failed.

#### Scenario: Await compute operation

- **GIVEN** a submitted compute operation
- **WHEN** the Component or Runtime awaits completion
- **THEN** the operation reaches either a completed, cancelled, or failed
  terminal state

#### Scenario: Cancel operation

- **GIVEN** a submitted compute operation
- **WHEN** cancellation is requested
- **THEN** the Runtime forwards cancellation to the selected Provider when
  safe cancellation is supported
- **AND** unsupported cancellation is reported as a stable structured error

---

### Requirement: Structured Compute Errors

Compute execution SHALL return stable structured errors.

Backend-specific diagnostic messages MAY be included as optional diagnostics.

Backend diagnostics SHALL NOT define the stable contract.

#### Scenario: Backend failure

- **GIVEN** a Provider returns a hardware-specific error
- **WHEN** the Runtime reports the failure
- **THEN** the error is mapped to a stable compute error code with optional
  diagnostic text

---

### Requirement: Compute Boundary Exclusions

The Compute WIT contract SHALL exclude backend-specific execution details,
including raw device handles, raw provider handles, GPU pointers, Candle
`Tensor`, Candle `BackendStorage`, Rust trait objects, backend-specific kernel
names, autograd state, and training behavior.

#### Scenario: Review portable contract surface

- **WHEN** a reviewer inspects the Compute WIT contract
- **THEN** no excluded native handle, Rust object, eager primitive catalog,
  autograd state, or training behavior is part of the stable WIT surface

### Requirement: Compute Operation Catalog

`magnetar:compute/run` SHALL define a Compute Operation Catalog.

The catalog SHALL group compute work into operation families.

Operation families SHALL describe coverage areas inside the Compute
Capability.

Operation families SHALL NOT automatically become separate Capabilities.

#### Scenario: Register operation family

Given an operation family is added to the catalog

When the Runtime loads the Compute Capability metadata

Then the family is available for validation and Provider advertisement.

---

### Requirement: Descriptor and View Operations

The catalog SHALL include descriptor and view operations.

Descriptor and view operations MAY include:

- shape inspection
- dtype inspection
- reshape
- flatten
- squeeze
- unsqueeze
- transpose
- permute
- narrow
- slice
- broadcast

These operations SHALL describe tensor metadata and view transformations.

They SHALL NOT expose native storage aliases or backend layout objects.

#### Scenario: Validate view operation

Given a submitted compute graph contains a view operation

When the Runtime validates the graph

Then the Runtime validates shape, dtype and view constraints before execution.

---

### Requirement: Construction and Allocation Requests

The catalog SHALL include construction and allocation request operations.

These operations MAY include:

- scalar construction
- zero-filled tensor request
- one-filled tensor request
- range-like tensor request
- descriptor-based allocation request

Allocation policy, memory pools and unsafe initialization SHALL remain native
Provider or Runtime responsibilities.

#### Scenario: Request tensor allocation

Given a compute graph requests a tensor allocation

When the Runtime validates the request

Then the request is checked against descriptor limits before Provider execution.

---

### Requirement: Data Movement and Conversion Operations

The catalog SHALL include data movement and conversion operations.

These operations MAY include:

- upload
- download
- copy
- materialize
- dtype conversion
- placement transfer

Data movement SHALL be explicit.

The Runtime SHALL NOT hide cross-Provider transfer or CPU staging behind
implicit execution behavior.

#### Scenario: Transfer tensor resource

Given a tensor resource belongs to one Provider or Device

When another Provider or Device needs to consume it

Then the Runtime requires an explicit transfer, copy or materialization step.

---

### Requirement: Elementwise Operations

The catalog SHALL include elementwise operations.

Elementwise operations MAY include:

- arithmetic operations
- unary mathematical operations
- activation functions
- affine transforms
- power functions

Rust operator traits and scalar generics SHALL NOT be part of the portable
contract.

#### Scenario: Execute elementwise operation

Given a compute graph contains an elementwise operation

When the selected Provider advertises support for that operation family

Then the Provider may execute the operation using its native kernels.

---

### Requirement: Comparison and Selection Operations

The catalog SHALL include comparison and selection operations.

Comparison and selection operations MAY include:

- equality comparison
- ordering comparison
- conditional selection
- where-like selection

#### Scenario: Validate selection operation

Given a compute graph contains a conditional selection operation

When the Runtime validates the graph

Then input compatibility and output descriptor rules are checked before
execution.

---

### Requirement: Reduction Operations

The catalog SHALL include reduction operations.

Reduction operations MAY include:

- sum
- mean
- minimum
- maximum
- argmin
- argmax

Reduction semantics SHALL later specify axis behavior, keep-dimension behavior,
dtype behavior and empty-input behavior.

#### Scenario: Execute reduction

Given a compute graph contains a reduction operation

When the Runtime validates the graph

Then axes, output shape and dtype constraints are checked.

---

### Requirement: Linear Algebra Operations

The catalog SHALL include linear algebra operations.

Linear algebra operations MAY include:

- matrix multiplication
- batched matrix multiplication
- broadcast matrix multiplication

Concrete schemas SHALL later specify batching, transpose behavior,
accumulation dtype, precision policy and quantization interaction.

#### Scenario: Execute matrix multiplication

Given a compute graph contains a matrix multiplication operation

When the selected Provider advertises compatible linear algebra support

Then the Runtime may submit the operation to that Provider.

---

### Requirement: Convolution and Spatial Transform Operations

The catalog SHALL include convolution and spatial transform operations.

These operations MAY include:

- convolution
- transposed convolution
- pooling
- nearest upsampling
- bilinear upsampling

Concrete schemas SHALL later specify layout, padding, dilation, stride,
coordinate and numerical semantics.

#### Scenario: Validate convolution operation

Given a compute graph contains a convolution operation

When the Runtime validates it

Then descriptor, layout and parameter constraints are checked before execution.

---

### Requirement: Indexing and Update Operations

The catalog SHALL include indexing and update operations.

These operations MAY include:

- gather
- index select
- index add
- scatter
- scatter add
- slicing
- concatenation

Mutation and aliasing SHALL be represented with explicit result or resource
semantics at WIT boundaries.

#### Scenario: Execute indexing operation

Given a compute graph contains an indexing operation

When the Runtime validates it

Then input index descriptors and output descriptors are checked.

---

### Requirement: Random Generation Operations

The catalog SHALL include random generation operations.

Random generation operations MAY include:

- uniform distribution
- normal distribution
- explicit optional seed
- generation into an opaque tensor resource

The Runtime SHALL NOT assume bitwise deterministic results across Providers.

#### Scenario: Generate random tensor

Given a compute graph requests random generation

When no seed is specified

Then the selected Provider may choose its own randomness policy.

---

### Requirement: Synchronization and Completion

The catalog SHALL include operation completion semantics.

Components SHALL await coarse operations or sessions.

Components SHALL NOT synchronize hardware queues directly.

#### Scenario: Await submitted work

Given a compute operation has been submitted

When the caller awaits completion

Then the Runtime returns a completed, cancelled or failed terminal state.

---

### Requirement: Provider Operation Advertisement

Providers SHALL advertise supported operation families for `magnetar:compute/run`.

Provider advertisements MAY include:

- supported operation families
- supported dtypes
- supported descriptor limits
- supported layout constraints
- supported precision modes

#### Scenario: Select Provider

Given multiple Providers implement `magnetar:compute/run`

When the Runtime resolves a compute request

Then operation family support is considered during Provider selection.

---

### Requirement: Unsupported Operation Handling

The Runtime SHALL reject unsupported operation families before Provider
execution begins when support cannot be resolved.

#### Scenario: Unsupported operation family

Given a compute graph contains an operation family unsupported by all compatible
Providers

When the Runtime validates the graph

Then it returns a structured unsupported-operation-family error.

---

### Requirement: Initial Exclusions

The initial Compute Operation Catalog SHALL exclude:

- autograd
- training graphs
- arbitrary Rust custom operations
- backend-specific kernel names
- raw backend storage operations
- direct Component access to hardware queues

#### Scenario: Reject excluded operation

Given a compute graph requests an excluded operation

When the Runtime validates the graph

Then it returns a structured unsupported-operation error.

