# Stabilize Compute Run Boundary Design

## Context

`magnetar:compute/run@1.0.0` is a marker contract. The archived Compute design
intentionally left mathematical operations for a later compatible revision once
their semantics were specified.

The capability taxonomy then identified Candle tensor and backend APIs as
evidence for a hybrid boundary: descriptors can cross WIT, while allocation,
storage, kernel dispatch, queues, synchronization, and native backend handles
remain Provider-owned.

## Goals / Non-Goals

Goals:

- Introduce the first executable `magnetar:compute/run` WIT surface as version
  `1.1.0`.
- Keep the boundary coarse by submitting opaque graph resources rather than
  exposing one WIT call per tensor primitive.
- Define portable fixed-width descriptors for tensor metadata.
- Define opaque tensor, graph, and operation resources with explicit lifecycle
  and completion semantics.
- Return stable structured errors while preserving optional backend diagnostics.

Non-Goals:

- Define graph construction, operation catalogs, or numerical semantics.
- Define data upload, download, copy, memory planning, or placement transfer.
- Expose raw Provider, Device, GPU, Candle, Rust trait, or backend storage
  handles in WIT.
- Define autograd, training, or custom-kernel extension behavior.

## Decisions

### Version the executable contract as 1.1.0

The `1.0.0` contract remains the initial marker. Adding concrete WIT functions
and resources is a new provider obligation, so Magnetar advertises it as
`magnetar:compute/run@1.1.0` rather than rewriting the historical marker.

This keeps the same stable major version and matches the runtime's existing
semantic compatibility rule: an available `1.1.0` capability can satisfy a
`1.0.0` request, while a provider that only supports the marker does not claim
the executable surface.

Alternative: make the new surface `2.0.0`. This was rejected because the
Capability identity and high-level role do not change, and the existing marker
did not define behavior that would be broken by the new executable boundary.

### Use one coarse run interface

The `run` interface owns submission and operation lifecycle. The initial call
accepts a borrowed opaque graph plus borrowed input tensors and returns an
operation resource. Graph construction and data movement are deferred, so the
resources are intentionally not constructible in this interface.

Alternative: split tensors, graphs, and execution into separate interfaces
now. This was rejected because the current runtime derives Capability identity
from the imported WIT interface name, so splitting interfaces would accidentally
create separate Capability IDs before their dependency and affinity rules are
specified.

### Keep tensor storage and graph internals opaque

Tensor descriptors expose shape, dtype, and view metadata using fixed-width
integer types. Tensor storage, aliasing internals, graph representation, native
queues, and kernel dispatch remain Provider-owned.

This mirrors the taxonomy's Candle finding: `Tensor`, `Storage`, and
`BackendStorage` motivate the boundary but are not copied into it.

### Model operations as Provider-pinned resources

A submitted operation is pinned to the Provider that owns its graph and tensor
resources. Fallback remains transparent only before these resources exist. Once
an operation is submitted, cancellation or failure is surfaced through the
operation lifecycle; the Runtime does not silently migrate live state.

Dropping a resource handle releases the Component's reference. It does not
guarantee cancellation unless the Component explicitly calls `cancel`.

### Use stable error codes plus diagnostics

`compute-error` contains a stable code and an optional diagnostic string.
Provider-specific error messages can aid debugging, but they do not define
portable behavior. Components branch on the stable code.

## Risks / Trade-offs

- Graph and tensor resources are not constructible in this first interface.
  This is deliberate: construction, data movement, and operation catalogs need
  separate semantics before they become portable.
- `await-completion` is blocking at the WIT surface. A future async or polling
  integration can wrap the same operation resource without changing the
  terminal states.
- Cancellation is best-effort because not every Provider can safely preempt
  backend work. Unsupported cancellation returns a stable error instead of
  pretending the operation was stopped.

## Migration Plan

Update the WIT package and canonical runtime metadata to `1.1.0`. Providers
that implement the executable surface advertise `compute_capability()`.
Providers that only implemented the marker must keep advertising `1.0.0` or
remain unregistered for this Capability until they implement the new contract.
