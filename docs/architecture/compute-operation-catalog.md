# Compute Operation Catalog

`magnetar:compute/run` is the portable Compute Capability boundary. The
Compute Operation Catalog documents the semantic families of work that may be
submitted through that coarse boundary.

Operation families are not separate Capabilities. They are validation and
advertisement metadata inside the Compute Capability, so Providers can state
which parts of Compute they support without fragmenting Component imports.

The initial catalog contains:

- descriptor and view operations
- construction and allocation requests
- data movement and conversion
- elementwise operations
- comparison and selection operations
- reduction operations
- linear algebra operations
- convolution and spatial transform operations
- indexing and update operations
- random generation operations
- synchronization and completion semantics

Providers advertise supported operation families together with optional dtype,
layout and precision constraints. The Runtime validates submitted compute work
against known operation families and the selected Provider's advertisement
before Provider execution begins.

## Tensor Descriptor Model

Tensor descriptors carry portable metadata only. A descriptor contains a
fixed-width shape, a dtype descriptor, a layout descriptor, and optional view
metadata. Dimensions, strides, offsets, element counts and byte-size
calculations use explicit `u64` or `i64` values at the boundary; `usize` and
other platform-sized assumptions are not part of the contract.

The initial portable dtype set is finite: bool, unsigned and signed 8/16/32/64
bit integers, f16, bf16, f32 and f64. Provider-specific dtypes may be named
explicitly, but they are valid only when the selected Provider advertises
support and their element size is known for descriptor validation.

Layouts are represented as contiguous, portable strided, or Provider-opaque.
Strided descriptors validate stride rank, offset bounds and span overflow
before execution. Provider-opaque layouts identify a Provider-managed layout
constraint without exposing native layout objects.

Tensor storage is represented by opaque tensor resources. Host-side resource
descriptors pair portable tensor metadata with `ResourceAffinity`, preserving
the selected Provider, Device and Compute capability binding. Components may
pass resources through compatible calls, but they cannot inspect raw storage,
native tensor objects, backend handles, GPU pointers, queues, locks or
Provider-owned memory through WIT.

Views are descriptors that depend on an existing descriptor or tensor resource.
They are not independent storage. A materialized copy is represented as a new
opaque tensor resource with its own affinity, so cross-Provider use requires an
explicit transfer, copy or materialization step.

The initial catalog deliberately excludes autograd, training graphs, arbitrary
Rust custom operations, backend-specific kernel names, raw backend storage and
direct Component access to hardware queues. Those concerns either remain native
Provider or Runtime implementation details, or require later Capability-level
design work.

Future changes may define operation-specific schemas, numerical semantics,
axis and layout behavior, transfer behavior, quantization interaction and exact
WIT representation. Until then, `ComputeOperationDescriptor` is a placeholder
that carries only the family, broad compatibility constraints and portable
tensor descriptors. Magnetar does not adopt Candle's eager Tensor semantics:
Candle's Rust `Tensor`, `Shape`, `Layout`, storage enums, locks, autograd graph
and backend dispatch APIs remain evidence for the boundary, not the boundary
itself.
