## ADDED Requirements

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
