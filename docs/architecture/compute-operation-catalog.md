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

The initial catalog deliberately excludes autograd, training graphs, arbitrary
Rust custom operations, backend-specific kernel names, raw backend storage and
direct Component access to hardware queues. Those concerns either remain native
Provider or Runtime implementation details, or require later Capability-level
design work.

Future changes may define operation-specific schemas, numerical semantics,
axis and layout behavior, transfer behavior, quantization interaction and exact
WIT representation. Until then, `ComputeOperationDescriptor` is a placeholder
that carries only the family and broad compatibility constraints.
