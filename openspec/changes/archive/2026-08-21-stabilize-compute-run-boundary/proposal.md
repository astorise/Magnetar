# Stabilize Compute Run Boundary

## Why

The capability taxonomy identified `magnetar:compute/run` as the first
Compute-related boundary that should evolve from a marker into an executable
WIT-backed contract.

Candle provides evidence for tensor descriptors, storage, layouts, dtype,
operations, and backend dispatch, but its eager `Tensor` API and backend
storage types are not portable WIT contracts.

Magnetar must keep Components away from raw Provider and Device handles while
also avoiding one WIT function per eager tensor primitive. Components should
submit coarse compute work to the Runtime through opaque graph and tensor
resources owned by the selected Provider.

## What Changes

This proposal evolves `magnetar:compute/run` to version `1.1.0` as a coarse
compute submission Capability. The previous `1.0.0` marker remains the
historical initial contract; the executable surface is introduced as a minor
version because it keeps the same Capability identity and major version.

The contract boundary includes:

- fixed-width tensor, shape, dtype, and view descriptors
- opaque tensor and graph resources
- an operation resource for submitted compute work
- submit, await, cancel, status, and output retrieval semantics
- stable structured compute errors with optional backend diagnostics

The contract boundary excludes:

- raw Provider or Device handles
- GPU pointers or backend storage objects
- Rust trait objects, generics, or Candle-specific types
- eager per-operation WIT calls
- backend-specific kernel names
- autograd or training state

Data movement, memory planning, graph construction, operation catalogs, and
numerical semantics will be specified by future changes.

## Capabilities

New Capabilities:

- None

Modified Capabilities:

- `capability`: evolves `magnetar:compute/run` from the `1.0.0` marker into a
  `1.1.0` coarse graph execution boundary.

## Impact

Affected artifacts:

- `openspec/specs/capability/spec.md`
- `magnetar-runtime/wit/compute.wit`
- `magnetar-runtime/src/lib.rs`
- `docs/architecture/capability-taxonomy.md`

Providers that advertise `magnetar:compute/run@1.1.0` must implement the
stable WIT surface defined by this change. Existing compatibility rules still
allow a `1.1.0` provider to satisfy a `1.0.0` request for the same Capability
major version.
