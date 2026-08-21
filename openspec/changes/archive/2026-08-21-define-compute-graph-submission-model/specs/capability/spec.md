## ADDED Requirements

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
