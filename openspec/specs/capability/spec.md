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

### Requirement: Tensor Descriptor

Magnetar SHALL define a portable Tensor Descriptor model.

A Tensor Descriptor SHALL describe tensor metadata.

A Tensor Descriptor SHALL NOT expose native storage, backend handles, locks,
queues, streams, GPU pointers or Rust objects.

#### Scenario: Describe tensor

Given a Component needs to submit tensor work

When it constructs a Tensor Descriptor

Then the descriptor contains portable metadata only.

---

### Requirement: Shape Descriptor

A Tensor Descriptor SHALL include a Shape Descriptor.

A Shape Descriptor SHALL represent tensor rank and dimensions using fixed-width
integer values.

Shape dimensions SHALL NOT use platform-sized integer types.

#### Scenario: Validate shape

Given a Tensor Descriptor with shape metadata

When the Runtime validates the descriptor

Then the Runtime checks rank, dimensions and element count constraints before
Provider execution.

---

### Requirement: Shape Overflow Validation

The Runtime SHALL validate tensor size calculations for overflow.

Validation SHALL include:

- rank limit
- dimension limit
- element count overflow
- byte-size overflow
- Provider-supported maximum size

#### Scenario: Reject overflowing shape

Given a Tensor Descriptor whose dimensions overflow the supported element count

When the Runtime validates it

Then the Runtime rejects the descriptor with a structured invalid-shape error.

---

### Requirement: DType Descriptor

A Tensor Descriptor SHALL include a DType Descriptor.

The DType Descriptor SHALL use stable portable dtype identifiers.

Providers SHALL advertise supported dtypes.

#### Scenario: Unsupported dtype

Given a Tensor Descriptor uses a dtype unsupported by all compatible Providers

When compute work is validated

Then the Runtime rejects the request with a structured unsupported-dtype error.

---

### Requirement: Initial DType Set

The initial DType Descriptor model SHALL support an explicit finite set of
portable dtype identifiers.

The initial dtype set MAY include:

- bool
- u8
- u16
- u32
- u64
- i8
- i16
- i32
- i64
- f16
- bf16
- f32
- f64

Provider-specific, quantized or experimental dtypes SHALL require explicit
advertisement before use.

#### Scenario: Experimental dtype

Given a Tensor Descriptor uses a Provider-specific dtype

When the Provider has not advertised support for that dtype

Then the Runtime rejects the descriptor.

---

### Requirement: Layout Descriptor

A Tensor Descriptor SHALL define a Layout Descriptor model and MAY omit layout
metadata when no layout constraint is required.

The Layout Descriptor SHALL describe portable layout constraints.

The Layout Descriptor SHALL NOT expose backend-specific layout objects.

#### Scenario: Validate layout

Given a Tensor Descriptor includes layout metadata

When the Runtime validates it

Then the Runtime checks that the selected Provider supports the requested
layout constraints.

---

### Requirement: Contiguous Layout

The Tensor Descriptor model SHALL support contiguous layout as a portable
layout kind.

#### Scenario: Contiguous tensor

Given a Tensor Descriptor requires contiguous layout

When a Provider receives the request

Then the Provider either accepts the contiguous constraint or rejects it with a
structured unsupported-layout error.

---

### Requirement: Strided Layout

The Tensor Descriptor model SHALL support portable strided layout when a
portable view requires explicit strides.

Strides and offsets SHALL use fixed-width integer values.

Strided layout SHALL NOT imply direct access to native storage.

#### Scenario: Strided view

Given a Tensor Descriptor describes a strided view

When the Runtime validates the descriptor

Then stride, offset and bounds constraints are checked before execution.

---

### Requirement: Opaque Tensor Resource

Tensor storage SHALL be represented as an opaque Tensor Resource.

Components MAY pass Tensor Resources between compatible calls.

Components SHALL NOT inspect or mutate Tensor Resource storage directly.

#### Scenario: Pass tensor resource

Given a Component receives an opaque Tensor Resource

When it submits compute work using that resource

Then the Runtime validates the resource affinity and descriptor compatibility
before Provider execution.

---

### Requirement: Tensor Resource Affinity

Every Tensor Resource SHALL carry Resource Affinity metadata.

The affinity metadata SHALL record Provider and Device binding when applicable.

#### Scenario: Use tensor across Providers

Given a Tensor Resource is bound to one Provider

When another Provider attempts to consume it

Then the Runtime requires an explicit transfer, copy or materialization step
before execution.

---

### Requirement: View Descriptor

The Tensor Descriptor model SHALL distinguish tensor views from materialized
tensor copies.

A View Descriptor SHALL describe how a tensor view relates to its source
resource or descriptor.

#### Scenario: Create view

Given a tensor view is created from another tensor resource

When the view is represented across the WIT boundary

Then the Runtime records that the view depends on the source resource rather
than treating it as an independent materialized copy.

---

### Requirement: Materialized Copy

A materialized tensor copy SHALL be represented as a distinct Tensor Resource.

#### Scenario: Materialize view

Given a tensor view cannot be consumed by a selected Provider

When materialization is explicitly requested

Then the Runtime creates a distinct Tensor Resource with its own affinity.

---

### Requirement: Descriptor Validation Before Execution

The Runtime SHALL validate Tensor Descriptors before Provider execution begins.

#### Scenario: Invalid descriptor

Given a Tensor Descriptor is malformed or unsupported

When compute work is submitted

Then the Runtime rejects the request before invoking the Provider.

---

### Requirement: No Autograd Metadata

Tensor Descriptors SHALL NOT include autograd, training graph or gradient
metadata.

#### Scenario: Training metadata supplied

Given a Tensor Descriptor includes training-specific metadata

When the Runtime validates the descriptor

Then the Runtime rejects the metadata as unsupported.

---

### Requirement: Structured Tensor Descriptor Errors

The Runtime SHALL return stable structured errors for Tensor Descriptor
validation failures.

Structured errors SHALL include categories for:

- invalid shape
- invalid dtype
- invalid layout
- unsupported dtype
- unsupported layout
- size overflow
- incompatible resource affinity

Backend diagnostics MAY be attached for debugging but SHALL NOT define the
stable contract.

#### Scenario: Report descriptor failure

Given descriptor validation fails

When the Runtime reports the error

Then the error uses a stable structured Tensor Descriptor error variant.

### Requirement: Compute Graph Submission

`magnetar:compute/run` SHALL support coarse compute graph submission.

A Compute Graph SHALL represent a graph, batch or equivalent coarse unit of
compute work.

Components SHALL NOT call one WIT function per eager tensor primitive.

#### Scenario: Submit graph

Given a Component has compute work to execute

When it calls `magnetar:compute/run`

Then the work is submitted as a Compute Graph or equivalent coarse unit.

---

### Requirement: Compute Graph

A Compute Graph SHALL contain compute nodes, graph inputs and graph outputs.

A Compute Graph SHALL be portable across compatible Providers.

A Compute Graph SHALL NOT contain native handles, backend kernel names, Rust
objects, pointers, queues or streams.

#### Scenario: Validate graph shape

Given a Compute Graph

When the Runtime validates it

Then the Runtime verifies graph inputs, graph nodes and graph outputs before
Provider execution.

---

### Requirement: Compute Node

A Compute Node SHALL describe one semantic compute operation.

A Compute Node SHALL reference an operation family from the Compute Operation
Catalog.

A Compute Node SHALL reference inputs and outputs using portable graph values.

#### Scenario: Validate node operation family

Given a Compute Node references an operation family

When the Runtime validates the graph

Then the Runtime checks that the operation family is known and supported by at
least one compatible Provider.

---

### Requirement: Compute Values

A Compute Graph SHALL represent intermediate values using portable Compute
Values.

Compute Values MAY represent:

- graph inputs
- node outputs
- constants
- opaque tensor resources
- tensor descriptors

Compute Values SHALL NOT expose native storage.

#### Scenario: Use tensor resource input

Given a Compute Graph input references an opaque Tensor Resource

When the Runtime validates the graph

Then the Runtime checks descriptor compatibility and Resource Affinity before
Provider execution.

---

### Requirement: Tensor Descriptor Integration

Compute Graph inputs and outputs SHALL use the Tensor Descriptor Model.

Tensor descriptors SHALL be validated before Provider execution.

#### Scenario: Invalid tensor descriptor

Given a Compute Graph contains an invalid Tensor Descriptor

When the Runtime validates the graph

Then graph submission fails before Provider execution begins.

---

### Requirement: Resource Affinity Validation

The Runtime SHALL validate Resource Affinity for all opaque resources used by a
Compute Graph.

Provider-pinned resources SHALL only be consumed by compatible Providers unless
an explicit transfer, copy, materialization or replay step exists.

#### Scenario: Incompatible tensor resource

Given a Compute Graph references a Tensor Resource bound to one Provider

When the selected Provider is incompatible with that resource

Then the Runtime rejects submission or requires an explicit data movement step.

---

### Requirement: Provider Capability Validation

The Runtime SHALL validate that the selected Provider supports every required
operation family, dtype, layout and descriptor constraint.

#### Scenario: Unsupported operation

Given a Compute Graph contains an operation unsupported by the selected Provider

When the Runtime validates Provider compatibility

Then the Runtime rejects the graph with a structured unsupported-operation
error.

---

### Requirement: Graph Acyclicity

A Compute Graph SHALL be acyclic unless a future control-flow contract
explicitly defines cyclic semantics.

#### Scenario: Cyclic graph

Given a Compute Graph contains a cycle

When the Runtime validates it

Then the Runtime rejects it with a structured invalid-graph error.

---

### Requirement: Explicit Graph Inputs

A Compute Graph SHALL declare all external inputs explicitly.

External inputs MAY include tensor resources, tensor descriptors or constants.

#### Scenario: Missing input

Given a Compute Node references an undeclared graph input

When the Runtime validates the graph

Then the Runtime rejects it with a structured missing-input error.

---

### Requirement: Explicit Graph Outputs

A Compute Graph SHALL declare all observable outputs explicitly.

Outputs MAY become opaque Tensor Resources.

Produced Tensor Resources SHALL carry Resource Affinity metadata.

#### Scenario: Produce tensor output

Given a Compute Graph declares a tensor output

When execution completes successfully

Then the Runtime returns an opaque Tensor Resource with descriptor and affinity
metadata.

---

### Requirement: Compute Submission Resource

A submitted Compute Graph SHALL produce a Compute Submission or operation
resource.

The submission resource SHALL expose completion state.

Completion states SHALL include:

- pending
- running
- completed
- cancelled
- failed

#### Scenario: Track submission

Given a Compute Graph has been submitted

When the caller queries the submission

Then the Runtime returns the current stable completion state.

---

### Requirement: Await Completion

The Runtime SHALL support awaiting Compute Submission completion.

#### Scenario: Await graph execution

Given a Compute Submission is running

When the caller awaits completion

Then the Runtime returns completed, cancelled or failed terminal state.

---

### Requirement: Cancellation

The Runtime SHALL support cancellation of Compute Submissions when the selected
Provider can safely cancel the underlying work.

Cancellation SHALL eventually produce a terminal state.

#### Scenario: Cancel graph execution

Given a Compute Submission is running

When cancellation is requested

Then the Runtime forwards the request to the selected Provider

And the submission eventually reaches cancelled, completed or failed state.

---

### Requirement: Structured Graph Errors

The Runtime SHALL return stable structured errors for graph submission and
validation failures.

Structured errors SHALL include categories for:

- invalid graph
- cyclic graph
- missing input
- missing output
- invalid tensor descriptor
- incompatible resource affinity
- unsupported operation
- unsupported dtype
- unsupported layout
- Provider unavailable
- execution failed
- cancelled

Backend diagnostics MAY be attached but SHALL NOT define the stable contract.

#### Scenario: Report graph validation failure

Given graph validation fails

When the Runtime reports the error

Then the error uses a stable structured graph error variant.

---

### Requirement: Provider-Owned Execution

Providers SHALL own native graph execution details.

Native graph execution details include:

- kernel selection
- command submission
- memory planning
- storage allocation
- device queues
- synchronization
- backend-specific optimization

#### Scenario: Execute graph on Provider

Given a Compute Graph has passed Runtime validation

When it is submitted to a Provider

Then the Provider executes it using native mechanisms without exposing those
mechanisms to the Component.

---

### Requirement: No Autograd or Training Graph

Compute Graph submission SHALL NOT include autograd or training graph behavior
in the initial contract.

#### Scenario: Submit training graph

Given a Compute Graph contains training-specific metadata

When the Runtime validates it

Then the Runtime rejects the graph as unsupported.

### Requirement: Explicit Data Movement

`magnetar:compute/run` SHALL model data movement explicitly.

The Runtime SHALL NOT silently upload, download, copy, transfer, materialize or
stage tensor data without an explicit data movement operation.

#### Scenario: Cross-device tensor use

Given a Tensor Resource is bound to one Device

When a compute graph requires the tensor on another Device

Then the Runtime requires an explicit transfer, copy or materialization operation
before execution.

---

### Requirement: Upload Operation

The data movement model SHALL include an Upload operation.

Upload SHALL create a Tensor Resource from host-provided data and a Tensor
Descriptor.

Upload SHALL validate the host buffer size against the Tensor Descriptor.

#### Scenario: Upload host data

Given host-owned tensor data

And a compatible Tensor Descriptor

When Upload is executed

Then the Runtime creates an opaque Tensor Resource with Resource Affinity
metadata.

---

### Requirement: Download Operation

The data movement model SHALL include a Download operation.

Download SHALL copy data from a Tensor Resource into a portable host-visible
representation.

Download SHALL NOT expose Provider-owned storage directly.

#### Scenario: Download tensor data

Given a Tensor Resource

When Download is executed

Then the Runtime returns host-visible data that matches the validated Tensor
Descriptor.

---

### Requirement: Copy Operation

The data movement model SHALL include a Copy operation.

Copy SHALL create a distinct Tensor Resource from an existing Tensor Resource.

Copy SHALL preserve semantic tensor contents.

#### Scenario: Copy tensor resource

Given a Tensor Resource

When Copy is executed

Then the Runtime returns a distinct Tensor Resource with its own Resource
Affinity metadata.

---

### Requirement: Materialize Operation

The data movement model SHALL include a Materialize operation.

Materialize SHALL convert a tensor view into a distinct Tensor Resource.

Materialize SHALL be explicit.

#### Scenario: Materialize tensor view

Given a Tensor Resource represents a view

When a selected Provider cannot consume that view directly

Then the Runtime requires an explicit Materialize operation before execution.

---

### Requirement: Transfer Operation

The data movement model SHALL include a Transfer operation.

Transfer SHALL move or copy tensor data between compatible Provider or Device
placements.

Transfer SHALL validate source affinity, destination constraints and Provider
support.

#### Scenario: Transfer between Providers

Given a Tensor Resource is owned by one Provider

And another Provider must consume it

When Transfer is requested

Then the Runtime validates that a supported transfer path exists before
execution.

---

### Requirement: DType Conversion

The data movement model SHALL support explicit dtype conversion.

DType conversion SHALL validate source dtype, target dtype and Provider support.

#### Scenario: Convert dtype

Given a Tensor Resource with dtype `f32`

When conversion to `f16` is requested

Then the Runtime validates that the selected Provider supports the conversion.

---

### Requirement: Placement Conversion

The data movement model SHALL support explicit placement conversion.

Placement conversion SHALL describe a requested movement between resource
placements such as host, Provider, Device or Affinity Group.

#### Scenario: Convert placement

Given a Tensor Resource has one placement

When another placement is requested

Then the Runtime validates that the conversion is explicit and supported.

---

### Requirement: Host Buffer Descriptor

Upload and Download operations SHALL use a portable Host Buffer Descriptor.

The Host Buffer Descriptor SHALL include stable byte length and encoding
metadata.

The Host Buffer Descriptor SHALL NOT expose raw pointers or native memory
handles.

#### Scenario: Validate host buffer

Given an Upload operation with a Host Buffer Descriptor

When the Runtime validates it

Then the Runtime checks byte length, dtype, shape and encoding compatibility.

---

### Requirement: Resource Affinity Preservation

Data movement operations SHALL preserve or create Resource Affinity metadata.

Produced Tensor Resources SHALL record their owning Provider and Device when
applicable.

#### Scenario: Produced resource affinity

Given a Transfer operation creates a Tensor Resource on another Device

When the operation completes

Then the produced Tensor Resource records the new Device affinity.

---

### Requirement: No Implicit CPU Staging

The Runtime SHALL NOT hide CPU staging as an implementation detail when it
changes observable placement, cost or synchronization behavior.

If CPU staging is required for a transfer, the operation SHALL be represented or
reported through diagnostics.

#### Scenario: CPU staging required

Given a Provider cannot transfer directly to another Provider

And the only available path uses host staging

When Transfer is requested

Then the Runtime either executes an explicit host-staged transfer or rejects the
request.

---

### Requirement: No Native Handle Exposure

Data movement operations SHALL NOT expose native handles.

Forbidden values include:

- raw pointers
- GPU pointers
- backend storage objects
- device queues
- streams
- locks
- Provider handles
- file descriptors used as native memory handles

#### Scenario: Inspect transfer result

Given a Component receives a Tensor Resource after Transfer

When it inspects portable metadata

Then it observes descriptors and stable affinity identifiers only.

---

### Requirement: Provider Advertisement

Providers SHALL advertise supported data movement operations.

Provider advertisements MAY include:

- supported upload paths
- supported download paths
- supported copy paths
- supported materialization behavior
- supported transfer paths
- supported dtype conversions
- supported layout conversions
- size limits

#### Scenario: Select Provider for movement

Given a data movement operation is requested

When the Runtime evaluates compatible Providers

Then Provider data movement support is considered before execution.

---

### Requirement: Incompatible Movement Rejection

The Runtime SHALL reject unsupported or unsafe data movement before Provider
execution begins.

#### Scenario: Unsupported transfer

Given a Tensor Resource is bound to one Provider

And no compatible transfer path exists to the requested Provider

When Transfer is requested

Then the Runtime rejects the operation with a structured unsupported-transfer
error.

---

### Requirement: Structured Data Movement Errors

The Runtime SHALL return stable structured errors for data movement failures.

Structured errors SHALL include categories for:

- invalid host buffer
- invalid tensor descriptor
- incompatible resource affinity
- unsupported upload
- unsupported download
- unsupported copy
- unsupported materialization
- unsupported transfer
- unsupported dtype conversion
- unsupported layout conversion
- size overflow
- execution failed
- cancelled

Backend diagnostics MAY be attached for debugging but SHALL NOT define the
stable contract.

#### Scenario: Report data movement failure

Given a data movement operation fails validation

When the Runtime reports the error

Then the error uses a stable structured data movement error variant.

### Requirement: Stable Compute Errors

`magnetar:compute/run` SHALL expose stable structured compute errors.

Compute errors SHALL use stable error categories.

Backend-specific error strings SHALL NOT define the stable contract.

#### Scenario: Provider returns native error

Given a Provider returns a backend-specific error

When the Runtime reports the failure

Then the Runtime maps it to a stable Compute Error category

And backend-specific details are attached only as diagnostics.

---

### Requirement: Error Phase

Every Compute Error SHALL identify the phase in which the error occurred.

Error phases SHALL include:

- validation
- resolution
- affinity-validation
- submission
- execution
- cancellation
- completion
- interruption

#### Scenario: Report validation error

Given graph validation fails

When the Runtime returns an error

Then the error phase is `validation`.

---

### Requirement: Validation Errors

The Compute Error Model SHALL include validation errors.

Validation errors SHALL include:

- invalid tensor descriptor
- invalid shape
- invalid dtype
- invalid layout
- size overflow
- invalid graph
- cyclic graph
- missing input
- missing output

#### Scenario: Invalid graph

Given a Compute Graph is malformed

When the Runtime validates it

Then the Runtime rejects it with a structured validation error before Provider
execution begins.

---

### Requirement: Unsupported Feature Errors

The Compute Error Model SHALL include unsupported feature errors.

Unsupported feature errors SHALL include:

- unsupported operation
- unsupported operation family
- unsupported dtype
- unsupported layout
- unsupported data movement
- unsupported transfer
- unsupported materialization
- unsupported conversion

#### Scenario: Unsupported dtype

Given a Compute Graph requires a dtype unsupported by all compatible Providers

When the Runtime validates Provider compatibility

Then the Runtime returns a structured unsupported dtype error.

---

### Requirement: Resolution Errors

The Compute Error Model SHALL include resolution errors.

Resolution errors SHALL include:

- no compatible Provider
- policy rejected Provider
- Provider unavailable
- Device unavailable
- Capability version mismatch

#### Scenario: No compatible Provider

Given no Provider can satisfy a compute request

When the Runtime resolves the request

Then the Runtime returns a structured no-compatible-provider error.

---

### Requirement: Resource Affinity Errors

The Compute Error Model SHALL include Resource Affinity errors.

Resource Affinity errors SHALL include:

- incompatible resource affinity
- Provider-pinned resource
- Device-bound resource
- artifact fingerprint mismatch
- affinity group mismatch

#### Scenario: Incompatible tensor resource

Given a Tensor Resource is bound to one Provider

And a Compute Graph is resolved to another incompatible Provider

When the Runtime validates the graph

Then the Runtime rejects the submission with a structured resource affinity
error.

---

### Requirement: Execution Errors

The Compute Error Model SHALL include execution errors.

Execution errors SHALL include:

- execution failed
- execution interrupted
- execution cancelled
- operation timeout
- out of memory
- resource exhausted

#### Scenario: Provider execution failure

Given a Provider fails during compute execution

When the Runtime reports the failure

Then the Runtime returns a stable execution error category.

---

### Requirement: Cancellation Errors

Cancellation SHALL be represented as a terminal execution outcome.

Cancellation SHALL NOT be reported as an unknown execution failure.

#### Scenario: Operation cancelled

Given a Compute Submission is running

When cancellation succeeds

Then the submission reaches a cancelled terminal state.

---

### Requirement: Interruption Errors

Interruption SHALL be distinct from cancellation.

Interruption means execution could not continue because of Provider, Device,
resource or runtime failure.

#### Scenario: Provider interrupted

Given a Provider-pinned operation is running

When the owning Provider becomes unavailable

Then the Runtime reports an interruption instead of silently resolving another
Provider.

---

### Requirement: Recovery Hints

The Compute Error Model SHALL define stable Recovery Hint categories.

Compute Errors MAY include Recovery Hints.

Recovery Hints SHALL be advisory.

Recovery Hints SHALL NOT imply that the Runtime has already retried, migrated or
replayed execution.

Supported Recovery Hints SHALL include:

- not-retryable
- retry-before-state
- restartable-with-replay
- explicit-transfer-required
- explicit-materialization-required
- Provider-pinned

When present, Recovery Hints SHALL use the stable hint categories defined by
this model.

#### Scenario: Provider-pinned failure

Given a Provider-pinned generation or compute session fails after state creation

When the Runtime reports the error

Then the error may include a Provider-pinned recovery hint.

---

### Requirement: No Automatic Migration Claim

The Compute Error Model SHALL NOT claim automatic live state migration.

Errors MAY describe whether work is transparent, restartable or Provider-pinned.

The Runtime SHALL NOT report a migrated result unless a future migration
contract explicitly defines it.

#### Scenario: Live state failure

Given a Provider-pinned resource has observable state

When the owning Provider fails

Then the Runtime reports interruption or failure

And does not claim successful migration.

---

### Requirement: Diagnostic Payload

The Compute Error Model SHALL define stable diagnostic payload rules.

Compute Errors MAY include diagnostic payloads.

When present, diagnostic payloads SHALL use stable identifiers and redacted
debug strings.

Diagnostics MAY include:

- Provider identifier
- Device identifier
- Capability identifier
- operation family
- rejected candidate identifiers
- backend diagnostic message
- debug trace identifier

Diagnostics SHALL NOT expose:

- raw pointers
- GPU pointers
- backend storage objects
- queues
- streams
- locks
- native handles
- credentials
- ambient filesystem paths

#### Scenario: Inspect diagnostic error

Given a Compute Error contains diagnostics

When a Component inspects the error

Then it observes stable identifiers and redacted diagnostic strings only.

---

### Requirement: Stable Error Serialization

Compute Errors SHALL be serializable across WIT boundaries.

Error fields SHALL use portable value types.

Error fields SHALL NOT contain Rust trait objects, associated types, callbacks,
channels or platform-specific handles.

#### Scenario: Return error through WIT

Given a compute request fails

When the error crosses the WIT boundary

Then it is represented using stable portable values.

---

### Requirement: Error Compatibility

Future versions of the Compute Error Model SHALL preserve compatibility for
existing stable error categories.

New error categories MAY be added in compatible versions when callers can treat
them as a general compute error.

#### Scenario: Add new error category

Given a future Compute Error category is introduced

When an older caller receives it

Then the caller can still handle it through a stable generic error fallback.

### Requirement: Compute Operation Schema Model

`magnetar:compute/run` SHALL define portable Compute Operation Schemas.

A Compute Operation Schema SHALL describe a graph node operation.

A Compute Operation Schema SHALL NOT be exposed as an individual WIT function.

A Compute Operation Schema SHALL NOT be registered as a separate Capability by
default.

#### Scenario: Validate operation schema

Given a Compute Graph contains a Compute Operation

When the Runtime validates the graph

Then the Runtime validates the operation against its Compute Operation Schema.

---

### Requirement: Operation Identifier

Every Compute Operation Schema SHALL have a stable operation identifier.

The identifier SHALL be portable and Provider-independent.

#### Scenario: Unknown operation

Given a Compute Graph contains an unknown operation identifier

When the Runtime validates the graph

Then validation fails with a structured unsupported-operation error.

---

### Requirement: Operation Family

Every Compute Operation Schema SHALL belong to one Compute Operation Family.

Operation families SHALL correspond to the Compute Operation Catalog.

#### Scenario: Validate operation family

Given a Compute Operation references a valid operation identifier

When the Runtime validates the operation

Then the Runtime verifies that the operation belongs to a known operation family.

---

### Requirement: Provider Operation Support

Providers SHALL advertise supported Compute Operation Schemas.

Provider support MAY vary by dtype, layout, shape limits, precision policy and
Device.

#### Scenario: Provider does not support operation

Given a Compute Graph contains a valid operation

And the selected Provider does not support that operation schema

When the Runtime validates Provider compatibility

Then validation fails with a structured unsupported-operation error.

---

### Requirement: Descriptor and View Operation Schemas

The initial schema set SHALL include descriptor and view operations.

The initial descriptor and view operation schemas SHALL include:

- `tensor.reshape`
- `tensor.transpose`
- `tensor.permute`
- `tensor.slice`
- `tensor.broadcast`
- `tensor.squeeze`
- `tensor.unsqueeze`

These operations SHALL transform tensor descriptors or views.

They SHALL NOT expose native storage aliases, raw strides or backend layout
objects.

#### Scenario: Reshape tensor

Given a `tensor.reshape` operation

When the Runtime validates it

Then the input and output element counts must be compatible.

---

### Requirement: View Versus Materialized Copy

View operation schemas SHALL preserve view semantics.

A view operation SHALL NOT imply a materialized copy.

Materialization SHALL require an explicit data movement operation.

#### Scenario: Provider cannot consume view

Given a Compute Graph produces a tensor view

And the selected Provider cannot consume that view

When the graph is validated

Then the Runtime requires explicit materialization or rejects execution.

---

### Requirement: Unary Elementwise Operation Schema

The initial schema set SHALL include a unary elementwise operation schema.

Unary elementwise operations SHALL consume one tensor-like input and produce one
tensor-like output.

The initial unary operator identifiers MAY include:

- `abs`
- `neg`
- `exp`
- `log`
- `sqrt`
- `recip`
- `sin`
- `cos`
- `tanh`
- `relu`
- `silu`
- `gelu`
- `erf`
- `floor`
- `ceil`
- `round`

#### Scenario: Unary elementwise validation

Given a unary elementwise operation

When the Runtime validates it

Then the input dtype, output dtype and Provider support are checked before
execution.

---

### Requirement: Binary Elementwise Operation Schema

The initial schema set SHALL include a binary elementwise operation schema.

Binary elementwise operations SHALL consume two tensor-like inputs and produce
one tensor-like output.

The initial binary operator identifiers MAY include:

- `add`
- `sub`
- `mul`
- `div`
- `maximum`
- `minimum`

Binary elementwise operations SHALL validate broadcasting rules.

#### Scenario: Binary broadcasting

Given a binary elementwise operation with two input tensors

When the Runtime validates it

Then the Runtime checks that their shapes are broadcast-compatible.

---

### Requirement: Comparison Operation Schema

The initial schema set SHALL include a comparison operation schema.

Comparison operations SHALL consume tensor-like inputs and produce boolean
tensor-like outputs unless a future schema explicitly defines another result
type.

The initial comparison operator identifiers MAY include:

- `eq`
- `ne`
- `lt`
- `le`
- `gt`
- `ge`

#### Scenario: Compare tensors

Given a comparison operation

When the Runtime validates it

Then input compatibility and boolean output descriptor rules are checked.

---

### Requirement: Conditional Selection Operation Schema

The initial schema set SHALL include a conditional selection operation schema.

The conditional selection operation SHALL represent where-like selection.

It SHALL consume:

- a boolean condition tensor
- a tensor-like true value
- a tensor-like false value

It SHALL produce one tensor-like output.

#### Scenario: Validate conditional selection

Given a conditional selection operation

When the Runtime validates it

Then condition shape, branch shape and output descriptor compatibility are
checked.

---

### Requirement: Reduction Operation Schema

The initial schema set SHALL include a reduction operation schema.

Reduction operations SHALL consume one tensor-like input and produce one
tensor-like output.

The initial reduction operator identifiers MAY include:

- `sum`
- `mean`
- `min`
- `max`
- `argmin`
- `argmax`

Reduction attributes SHALL include axes and keep-dimension behavior.

#### Scenario: Validate reduction axes

Given a reduction operation

When the Runtime validates it

Then axes are checked against the input rank.

---

### Requirement: Reduction Output Rules

Reduction schemas SHALL define output shape rules.

Reduction schemas SHALL define output dtype rules.

Empty-input behavior SHALL remain explicitly unresolved unless specified by a
future revision.

#### Scenario: Reduction output descriptor

Given a reduction operation with valid axes

When the Runtime validates it

Then the Runtime derives or checks the expected output descriptor.

---

### Requirement: Matrix Multiplication Schema

The initial schema set SHALL include a matrix multiplication schema.

Matrix multiplication SHALL support rank-2 matrix multiplication.

Batched matrix multiplication MAY be represented by a separate schema or by a
schema attribute.

Transpose behavior, accumulation dtype, precision policy and quantization
interaction SHALL be explicitly modeled before they are guaranteed stable.

#### Scenario: Validate matrix multiplication

Given a matrix multiplication operation

When the Runtime validates it

Then inner dimensions must be compatible.

---

### Requirement: Batched Matrix Multiplication Schema

The initial schema set SHALL include a batched matrix multiplication schema.

Batched matrix multiplication SHALL validate batch dimensions and matrix
dimensions separately.

#### Scenario: Validate batched matmul

Given a batched matrix multiplication operation

When the Runtime validates it

Then batch dimensions and matrix dimensions must be compatible.

---

### Requirement: Indexing Operation Schemas

The initial schema set SHALL include indexing operation schemas.

Indexing operation schemas MAY include:

- `tensor.gather`
- `tensor.index-select`
- `tensor.scatter`
- `tensor.scatter-add`

Index tensors SHALL use explicit supported integer dtypes.

Mutation SHALL be represented through explicit result resources rather than
implicit in-place mutation.

#### Scenario: Validate gather operation

Given a gather operation

When the Runtime validates it

Then index dtype, axis and output descriptor compatibility are checked.

---

### Requirement: Scatter Operation Semantics

Scatter-like operations SHALL define duplicate-index behavior before stable
cross-Provider equivalence is claimed.

#### Scenario: Duplicate scatter indices

Given a scatter operation with duplicate indices

When duplicate-index behavior is unspecified

Then the Runtime rejects the operation or marks it as requiring Provider-specific
semantics.

---

### Requirement: Concatenation Operation Schema

The initial schema set SHALL include a concatenation operation schema.

Concatenation SHALL consume a list of tensor-like inputs and produce one
tensor-like output.

The concatenation axis SHALL be explicit.

#### Scenario: Validate concatenation

Given a concatenation operation

When the Runtime validates it

Then all non-concatenated dimensions must be compatible.

---

### Requirement: Random Uniform Operation Schema

The initial schema set SHALL include a random uniform operation schema.

Random uniform SHALL produce a tensor-like output from a descriptor and
distribution parameters.

A seed MAY be provided explicitly.

Absence of a seed SHALL mean Provider-selected randomness.

#### Scenario: Random uniform without seed

Given a random uniform operation without a seed

When the operation executes

Then the selected Provider chooses its randomness policy.

---

### Requirement: Random Normal Operation Schema

The initial schema set SHALL include a random normal operation schema.

Random normal SHALL produce a tensor-like output from a descriptor and
distribution parameters.

The Runtime SHALL NOT assume bitwise deterministic results across Providers.

#### Scenario: Random normal with seed

Given a random normal operation with an explicit seed

When the operation executes on different Providers

Then reproducibility is only guaranteed if the selected Providers explicitly
advertise compatible deterministic behavior.

---

### Requirement: Operation Attribute Validation

Every operation schema SHALL define supported attributes.

The Runtime SHALL reject unknown or invalid attributes unless a schema explicitly
allows extension attributes.

#### Scenario: Unknown attribute

Given an operation contains an unknown attribute

When the Runtime validates it

Then validation fails with a structured invalid-operation-attribute error.

---

### Requirement: Operation Input Arity Validation

Every operation schema SHALL define input arity.

The Runtime SHALL reject operations with missing or extra inputs.

#### Scenario: Invalid arity

Given an operation declares the wrong number of inputs

When the Runtime validates it

Then validation fails with a structured invalid-operation-arity error.

---

### Requirement: Operation Output Validation

Every operation schema SHALL define output descriptor rules.

The Runtime SHALL validate declared outputs against schema rules.

#### Scenario: Invalid output descriptor

Given an operation declares an incompatible output descriptor

When the Runtime validates it

Then validation fails with a structured invalid-output-descriptor error.

---

### Requirement: Initial Schema Exclusions

The initial Compute Operation Schema set SHALL exclude:

- convolution
- pooling
- upsampling
- attention-specific fused operations
- normalization fused operations
- quantized operation schemas
- backend-specific kernel names
- arbitrary custom kernels
- autograd
- training graphs

#### Scenario: Excluded operation

Given a Compute Graph contains an excluded operation schema

When the Runtime validates it

Then validation fails with a structured unsupported-operation error.

---

### Requirement: Provider-Specific Extensions

Provider-specific operation extensions SHALL NOT be accepted as portable Compute
Operation Schemas unless they are explicitly declared as experimental or
Provider-specific.

Provider-specific extensions SHALL NOT be required by portable Components.

#### Scenario: Provider-specific operation

Given a Compute Graph contains a Provider-specific operation

When the Runtime validates it as portable compute

Then validation fails unless the operation is explicitly marked as a
Provider-specific extension and the selected Provider advertises support.

---

### Requirement: No Eager WIT Dispatch

Compute Operation Schemas SHALL be submitted inside a Compute Graph or
equivalent coarse unit.

Components SHALL NOT invoke each operation schema as a separate WIT call.

#### Scenario: Execute multiple operations

Given a Component needs to perform multiple tensor operations

When it uses `magnetar:compute/run`

Then the operations are submitted together as a graph, batch or equivalent coarse
unit.

