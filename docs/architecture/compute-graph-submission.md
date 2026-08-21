# Compute Graph Submission

`magnetar:compute/run` accepts compute work as a coarse graph, batch, or
equivalent unit. Components do not call one WIT function per eager tensor
primitive. The Runtime validates the submitted unit once, selects a compatible
Provider, and lets that Provider execute with native kernels, allocation,
queues and synchronization.

## Graph Model

A `ComputeGraph` has explicit inputs, ordered nodes and explicit observable
outputs. Inputs may describe tensors by portable `TensorDescriptor` values or
may reference opaque `TensorResource` values that already carry
`ResourceAffinity`.

Each `ComputeNode` names one semantic `ComputeOperationDescriptor`. The
descriptor references an operation family from the Compute Operation Catalog,
plus broad dtype, layout, precision and tensor descriptor constraints. It does
not expose backend kernel names or operation-specific native schemas.

Graph values are referenced through declared graph inputs or prior node
outputs. A node cannot depend on a future node output, which gives the initial
contract acyclic semantics. Cyclic or control-flow graphs require a later
contract that defines their execution behavior explicitly.

## Runtime Validation

Before invoking a Provider, the Runtime validates:

- graph, input, node and output identifiers
- input and output references
- acyclic node ordering
- tensor descriptor shape, dtype, layout, view and size constraints
- Provider support for every operation family
- Provider support for requested dtype, layout and precision constraints
- Resource Affinity compatibility for opaque tensor inputs

Validation errors are structured. Backend diagnostic text may be attached by
future adapters, but the stable contract is expressed through runtime error
categories such as invalid graph, missing input, cyclic graph, unsupported
operation family, unsupported dtype, unsupported layout and incompatible
resource affinity.

## Submission Lifecycle

A validated graph produces a `ComputeSubmission`. The submission records the
selected Provider, `magnetar:compute/run` capability version, execution
context and affinity group. The Provider owns native execution details after
validation; Components only observe lifecycle state and opaque output
resources.

Submission states are pending, running, completed, cancelled and failed. The
terminal states are completed, cancelled and failed. Cancellation is a request:
Providers may complete, cancel or fail the underlying work depending on what
can be safely interrupted.

Successful graph outputs are returned as opaque `TensorResourceDescriptor`
values. Produced tensor resources inherit the submission affinity so dependent
calls preserve Provider and Device ownership without exposing raw handles
through WIT.

## Relationship To The Compute Operation Catalog

The graph submission model is the container and lifecycle boundary. The
Compute Operation Catalog defines operation families that may appear inside
that container. Future changes can add operation-specific schemas and numerical
rules without changing the decision that WIT traffic remains coarse and
Provider-owned execution remains native.
