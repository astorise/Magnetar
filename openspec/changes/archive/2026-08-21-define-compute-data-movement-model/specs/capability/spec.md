## ADDED Requirements

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
