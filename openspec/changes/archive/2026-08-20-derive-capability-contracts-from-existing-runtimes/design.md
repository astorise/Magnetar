# Capability contract derivation design

## Context

Magnetar already models Providers, Devices, versioned Capabilities, portable
Components, deterministic Provider fallback, and the marker contract
`magnetar:compute/run@1.0.0`. It does not yet define tensor operations, model
sessions, generation, tokenization, or application-level AI contracts.

The analysis is reproducible against these upstream revisions:

- `huggingface/candle@2a13b0f3ff62f7e67013597f2996f764c5735e21`
- `lucasjinreal/Crane@a47b11ce9d36f269d3c100e1f84716b3dbf23777`

Candle is used as evidence for low-level device, storage, tensor, and module
boundaries. Crane is used as evidence for model generation, token streaming,
tokenization, and application abilities. Source APIs are evidence, not target
API definitions.

## Goals / Non-Goals

**Goals:**

- Produce a source-traceable taxonomy at low-level, model, and application
  layers.
- Map each responsibility to Magnetar's existing architectural roles.
- Record responsibility and dependency boundaries at a useful future contract
  granularity.
- Decide which boundaries stay native, can be portable Components, or require
  a hybrid of native resources and WIT control interfaces.
- Define fallback constraints before stateful contracts are standardized.

**Non-Goals:**

- Add concrete tensor operations to `magnetar:compute/run@1.0.0`.
- Add new Rust runtime APIs, Provider implementations, model loaders, or WASM
  engine integration.
- Guarantee final WIT package names or copy Candle and Crane interfaces.
- Treat every advertised or placeholder Crane ability as implemented behavior.

## Decisions

### Publish the result as architecture documentation

The implementation is a versioned architecture document rather than runtime
code. The change is a discovery gate for later contract changes, and exposing
provisional families through public Rust or WIT APIs would stabilize them too
early.

Alternative: add enums or constants for every family now. This was rejected
because the existing `CapabilityId` model already accepts package-qualified
identities and does not need a second provisional registry.

### Classify responsibilities before naming Capabilities

The taxonomy first maps each source responsibility to an existing Magnetar
role. A native-only responsibility remains owned by a Provider, Device, or
runtime service and is not called a Capability. Only a portable or hybrid WIT
boundary qualifies as a Capability candidate, preserving the existing rule
that every Capability exposes at least one WIT contract.

Alternative: call every backend responsibility a native Capability. This was
rejected because it conflicts with Magnetar's WIT-backed Capability model and
would blur the Provider/Capability boundary.

### Use three primary layers

Each responsibility family has one primary owner:

1. low-level execution owns device resources and mathematical execution;
2. model execution owns loaded models, token transforms, and inference sessions;
3. application abilities own user-visible request and result semantics.

Dependencies may cross upward between layers, but a family is not duplicated
merely because a higher-level ability uses it.

Alternative: mirror the source crate/module hierarchy. This was rejected
because source organization is implementation-specific and would reproduce
Candle and Crane coupling.

### Derive families by semantic and lifecycle cohesion

Operations belong together when they share data types, resource lifetime,
versioning pressure, and fallback behavior. Candle's large backend traits are
therefore evidence for several responsibility families rather than one copied
backend contract, while Crane's callbacks and channels become portable stream
events.

Alternative: create one Capability per source trait. This was rejected because
the traits mix responsibilities and expose Rust-only types and ownership.

### Keep provider-owned hot-path resources native

Device contexts, allocations, tensor storage, command submission, kernel
dispatch, synchronization primitives, loaded weights, and mutable inference
state stay behind the native Provider boundary. A future Component contract may
refer to them through opaque host resources and coarse operations, but it does
not own their raw representation.

Alternative: expose every tensor operation as an individual WIT call. This was
rejected because it would force fine-grained boundary crossings, complicate
zero-copy ownership, and leak backend scheduling details.

### Use WIT at coarse, portable semantic boundaries

Serializable request policy, token sequences, chat messages, stream events,
and application results are Component-suitable. Large tensors, model sessions,
audio, and image data require a hybrid design using opaque resources or streams
to avoid mandatory copies.

### Make fallback a property of state ownership

Provider selection is transparent only before Provider-owned state is created,
or for an idempotent operation whose complete input can be replayed before any
result becomes observable. Tensor handles, model sessions, random state,
incremental generation state, and active streams pin execution to their
Provider. Recovery requires an explicit restart and may not be deterministic.

## Risks / Trade-offs

- **Source drift** -> Pin revisions and record symbols so a later refresh is a
  deliberate change.
- **Taxonomy too broad** -> Record exclusions and split a family later when its
  semantics or versioning pressure diverge.
- **Premature package naming** -> Mark every WIT package as provisional and
  require a dedicated follow-up spec before registration.
- **Hidden data-copy costs** -> Classify large-data boundaries as hybrid and
  require opaque resources or streaming in future designs.
- **Misleading fallback promises** -> Distinguish pre-session selection,
  replayable restart, and Provider-pinned state.
- **Uneven source maturity** -> Mark placeholders and model-specific helpers as
  evidence only, not as guaranteed Magnetar functionality.

## Migration Plan

No runtime migration is required. Merge the taxonomy document and its README
link, then use separate OpenSpec changes to standardize selected WIT packages.
Removing this documentation change rolls back the implementation without
affecting runtime behavior.

## Open Questions

- Which tensor/graph representation should the first non-marker Compute
  revision accept?
- Which opaque resource types can be shared across separately versioned WIT
  packages without coupling their release cadence?
- What replay and determinism guarantees should a future scheduler require
  before attempting stateful fallback?
