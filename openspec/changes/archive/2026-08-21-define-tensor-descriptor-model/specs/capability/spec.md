## ADDED Requirements

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
