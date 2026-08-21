## ADDED Requirements

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
