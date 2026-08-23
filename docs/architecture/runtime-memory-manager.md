# Runtime Memory Manager

Magnetar treats memory as a Runtime-owned subsystem.

Compute describes portable tensor shape, dtype, layout, data movement, and
operation contracts. The Runtime Memory Manager owns the realization of those
descriptions into allocation requests, placement, residency, staging,
zero-copy feasibility, admission, pressure, and structured memory errors.

## Ownership

The Memory Manager is initialized by `Runtime` and exposed as a first-class
service. It is separate from:

- Compute descriptors and operation schemas
- Device metadata
- Provider execution internals
- Scheduler queueing
- Component Runtime APIs

Planning may consume Memory Manager decisions, but Planning does not own
allocator internals.

Providers may allocate native resources internally. Public Runtime and
Component-facing contracts use opaque Runtime identities, Resource Affinity,
and Memory Manager residency state instead of raw native memory handles.

## Allocation Model

The Memory Manager defines canonical allocation identity, class, placement,
owner, lifetime, state, admission, and release semantics.

Initial allocation classes include tensors, model artifacts, tokenizer
artifacts, adapters, quantization artifacts, KV cache, prefix cache, temporary
workspace, transfer staging, pinned host memory, browser linear memory, and
Runtime internal memory.

Initial placements include ordinary host memory, pinned host memory, device
memory, unified/shared memory, provider-owned opaque memory, external borrowed
memory, browser linear memory, and staged temporary memory.

## Caching Allocator

Released allocations may become reusable when cache policy permits it.
Compatible future allocation requests can reuse cached allocations by matching
class, placement, alignment, and size. Reuse emits cache-hit observations.

When no reusable allocation matches, the Memory Manager records a cache miss
and creates a new allocation. Cache limits are policy-governed; reusable
allocations over the configured cache budget are evicted and recorded as cache
eviction observations.

## Arenas

Memory arenas are explicit Runtime state. An arena has identity, allocation
class, placement, capacity, used bytes, owner, growth policy, shrink policy,
pressure, and diagnostics.

Fixed arenas reject reservations beyond capacity. Grow-on-demand arenas may
increase capacity by policy increments. Shrink policy can release reusable
capacity back to the current used size where applicable.

Arena pressure is derived from used bytes versus capacity and is emitted as an
arena pressure observation when reservations alter pressure.

## Pending Queues

Pending allocation queues are explicit. A pending allocation records request
identity, requested size, class, placement, affinity, priority, deadline,
queued time, cancellation state, and diagnostic reason.

Pending requests can be retried after pressure changes, cancelled by Runtime
policy, or failed when their deadline expires. Queue delay, queued allocation,
timeout, cancellation, and retry behavior are represented with structured
state and observations.

## Residency

Tensor residency is Runtime-owned state. Components cannot forge residency or
claim native placement directly.

Residency records connect tensor resource identity, memory allocation identity,
placement, and Resource Affinity. Provider-owned and staged residency are
represented explicitly.

Model, adapter, quantization artifact, KV cache, and prefix cache residency are
prepared as allocation classes and ownership categories. Their full domain
contracts remain future changes.

## Staging And Zero-Copy

Host staging is explicit. If a Compute data movement request forbids host
staging, the Memory Manager rejects staging instead of silently inserting it.

Zero-copy is a feasibility result with a stable reason. Matching placement can
be accepted; incompatible placement or storage/compute dtype mismatch is
rejected.

## Storage DType And Compute DType

The Memory Manager distinguishes storage dtype from compute dtype.

Storage dtype determines resident byte representation. Compute dtype determines
execution workspace requirements. Quantized or compressed storage can therefore
have a smaller storage footprint while still requiring temporary compute
workspace.

## Pressure And Admission

Memory pressure is represented separately from Provider failure. Saturated
memory pressure can reject or queue work according to Runtime policy without
marking a Provider failed solely because memory is constrained.

Admission decisions include admit, queue, reject, and retry-later. Decisions
carry stable reasons suitable for planning and scheduling diagnostics.

Scheduler consumes Runtime memory admission before accepting memory-dependent
work. A memory rejection becomes a stable memory-plan scheduling failure;
queue or retry-later decisions remain compatible with Scheduler queueing.

## Browser Constraints

Browser linear memory is an explicit placement. Native pinned memory is not
assumed for `wasm32` targets. Browser-specific memory limits and WebGPU buffer
semantics remain policy and Provider capability concerns.

## Observability

The Memory Manager records structured observations for allocation lifecycle,
release, cache, staging, zero-copy, and pressure categories. Observation
failures do not control memory decisions.
