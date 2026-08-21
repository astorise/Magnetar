## ADDED Requirements

### Requirement: Provider Compute Advertisement

Providers implementing `magnetar:compute/run` SHALL expose a Provider Compute
Advertisement.

The advertisement SHALL describe the Provider's portable compute support.

The advertisement SHALL NOT expose native handles, backend storage, kernel
symbols, queues, streams, locks or raw Device APIs.

#### Scenario: Register compute Provider

Given a Provider implements `magnetar:compute/run`

When the Provider is registered

Then the Runtime records its Provider Compute Advertisement.

---

### Requirement: Capability Version Support

A Provider Compute Advertisement SHALL declare the supported
`magnetar:compute/run` Capability versions.

#### Scenario: Resolve Capability version

Given a Component requires a specific `magnetar:compute/run` version

When the Runtime evaluates Providers

Then only Providers advertising a compatible version are considered.

---

### Requirement: Operation Family Support

A Provider Compute Advertisement SHALL declare supported Compute Operation
Families.

Operation Family support SHALL be used as a coarse compatibility signal.

#### Scenario: Evaluate operation family

Given a Compute Graph contains a linear algebra operation

When the Runtime evaluates a Provider

Then the Provider must advertise compatible linear algebra support.

---

### Requirement: Operation Schema Support

A Provider Compute Advertisement SHALL declare supported Compute Operation
Schemas.

Operation Schema support SHALL be more precise than Operation Family support.

#### Scenario: Unsupported operation schema

Given a Provider supports the linear algebra family

But does not support the requested matrix multiplication schema

When the Runtime validates Provider compatibility

Then the Runtime rejects that Provider for the graph.

---

### Requirement: Portable and Provider-Specific Operations

A Provider Compute Advertisement SHALL distinguish portable operation schemas
from Provider-specific extensions.

Provider-specific extensions SHALL NOT be required by portable Components.

#### Scenario: Provider-specific extension

Given a Compute Graph uses a Provider-specific operation

When the graph is validated as portable compute

Then validation fails unless the operation is explicitly marked as a
Provider-specific extension and the selected Provider advertises support.

---

### Requirement: DType Support

A Provider Compute Advertisement SHALL declare supported dtypes.

DType support MAY vary by operation schema, input position, output position and
Device.

#### Scenario: Unsupported dtype for operation

Given a Provider supports `tensor.matmul`

But only for `f16` and `f32`

When a Compute Graph requests `tensor.matmul` with `i8`

Then the Runtime rejects that Provider for the graph.

---

### Requirement: Layout Support

A Provider Compute Advertisement SHALL declare supported layout constraints.

Layout support MAY include:

- contiguous layout
- portable strided layout
- Provider-managed opaque layout
- view consumption support
- materialization requirement

#### Scenario: View requires materialization

Given a Tensor Resource is a view

And the selected Provider cannot consume that view directly

When the Runtime validates the graph

Then the Runtime requires explicit materialization or rejects execution.

---

### Requirement: Shape and Size Limits

A Provider Compute Advertisement SHALL declare shape and size limits.

Limits MAY include:

- maximum rank
- maximum dimension value
- maximum element count
- maximum byte size
- supported batch dimensions
- broadcasting constraints

#### Scenario: Shape exceeds Provider limit

Given a Compute Graph contains a Tensor Descriptor exceeding a Provider limit

When the Runtime validates Provider compatibility

Then the Runtime rejects that Provider before execution.

---

### Requirement: Precision Support

A Provider Compute Advertisement SHALL support declaring precision constraints.

Precision constraints MAY include:

- accumulation dtype
- approximate math support
- exact math support
- reduced precision support
- mixed precision support

#### Scenario: Precision policy required

Given a Compute Graph requires deterministic exact accumulation

When the Runtime evaluates a Provider

Then the Provider must advertise compatible precision support.

---

### Requirement: Determinism Support

A Provider Compute Advertisement SHALL support declaring deterministic behavior.

Determinism support SHALL be explicit.

The Runtime SHALL NOT assume bitwise equivalent results across Providers.

#### Scenario: Deterministic random generation

Given a Compute Graph requests deterministic random generation

When the Runtime evaluates Providers

Then only Providers advertising compatible deterministic random behavior are
eligible.

---

### Requirement: Data Movement Support

A Provider Compute Advertisement SHALL declare supported data movement paths
when the Provider participates in data movement.

Data movement support MAY include:

- upload
- download
- copy
- transfer
- materialize
- dtype conversion
- layout conversion
- host-staged transfer

#### Scenario: Transfer requires host staging

Given a transfer between two Providers requires host staging

When the Runtime evaluates the movement path

Then the Provider advertisement must indicate whether host staging is required
or unsupported.

---

### Requirement: Device-Specific Advertisement

A Provider Compute Advertisement SHALL support Device-specific compute support
when a Provider exposes different compute support for different Devices.

When support differs by Device, the Provider Compute Advertisement SHALL attach
constraints to stable Device identifiers.

#### Scenario: Multi-device Provider

Given one Provider exposes multiple Devices

And each Device has different memory or dtype support

When the Runtime evaluates candidates

Then Device-specific advertisement data is used during selection.

---

### Requirement: Advertisement and Resource Affinity

The Runtime SHALL evaluate Provider Compute Advertisements together with
Resource Affinity.

A Provider advertisement SHALL NOT override an existing Provider-pinned resource
affinity.

#### Scenario: Provider-pinned tensor

Given a Tensor Resource is bound to one Provider

And another Provider advertises support for the requested operation

When the Tensor Resource is used without explicit transfer

Then the Runtime rejects the second Provider for that operation.

---

### Requirement: Advertisement and Resolution Policy

Resolution Policies SHALL consider Provider Compute Advertisements.

A Provider that implements the requested Capability MAY still be rejected when
its advertisement does not satisfy the graph requirements.

#### Scenario: Capability compatible but graph incompatible

Given a Provider implements `magnetar:compute/run`

But does not advertise support for a required operation schema

When the Runtime resolves the graph

Then the Resolution Policy excludes that Provider.

---

### Requirement: Advertisement Validation

The Runtime SHALL validate Provider Compute Advertisements during Provider
registration.

Invalid advertisements SHALL prevent the affected support entry from being used.

#### Scenario: Invalid advertisement

Given a Provider advertises an unknown operation schema

When the Provider is registered

Then the Runtime rejects or ignores that advertisement entry with diagnostics.

---

### Requirement: Stable Advertisement Values

Provider Compute Advertisements SHALL use stable portable values.

Advertisement values SHALL NOT contain:

- Rust trait objects
- function pointers
- callbacks
- raw native handles
- backend object references
- platform-dependent integer assumptions

#### Scenario: Inspect advertisement

Given a Component or diagnostic tool inspects Provider support metadata

When advertisement data is returned

Then it contains only stable identifiers, versions, limits and portable values.

---

### Requirement: Structured Advertisement Errors

The Runtime SHALL return stable structured errors for advertisement-related
failures.

Structured errors SHALL include categories for:

- invalid advertisement
- unsupported operation schema
- unsupported operation family
- unsupported dtype
- unsupported layout
- unsupported precision policy
- unsupported deterministic behavior
- unsupported data movement
- Device constraint mismatch
- Resource Affinity conflict

Backend diagnostics MAY be attached but SHALL NOT define the stable contract.

#### Scenario: Report advertisement mismatch

Given no Provider advertisement satisfies a Compute Graph

When the Runtime reports the failure

Then it returns a stable structured advertisement or unsupported-feature error.

---

### Requirement: No Execution Guarantee

A Provider Compute Advertisement SHALL describe declared support.

It SHALL NOT guarantee successful execution.

Execution may still fail because of runtime conditions such as memory pressure,
Device unavailability, Provider interruption or resource exhaustion.

#### Scenario: Advertised operation fails at runtime

Given a Provider advertises support for an operation schema

When execution fails due to resource exhaustion

Then the Runtime reports a structured execution error rather than treating the
advertisement as false.
