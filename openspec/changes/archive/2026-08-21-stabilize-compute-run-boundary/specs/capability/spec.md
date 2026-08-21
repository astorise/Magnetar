## MODIFIED Requirements

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

## ADDED Requirements

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
