# Device Memory Pool And Allocation Planner

Magnetar separates logical memory policy from native allocation mechanics.

The Runtime Memory Manager owns logical Device memory pools, allocation
classes, reservations, reuse policy, pressure response, and admission. Providers
realize backing storage with native allocations or native pool facilities, but
native handles remain private. Devices expose capacity, pressure, and alignment
metadata; they are not allocator APIs.

## DeviceMemoryPool

`DeviceMemoryPool` is a Runtime-owned logical capacity domain bound to a
Provider, Device, memory domain, class, capacity, reservations, watermarks, and
state. A pool ID is opaque logical identity and must not encode native heap IDs,
vendor pool handles, pointers, file descriptors, or mapped addresses.

Baseline pool classes cover weights, KV cache, workspace, transient,
persistent, transfer, shared, and custom policy classes. The class names express
Runtime intent and do not mandate slab, buddy, arena, or vendor-specific
implementation.

## Reservations And Watermarks

`PoolCapacity` tracks configured limit, reserved bytes, committed bytes, leased
bytes, reclaimable bytes, pending reclaim bytes, and borrowed bytes.

Hard reservations protect capacity from unrelated lower-priority classes. Soft
reservations can be borrowed only through explicit policy and remain visible in
accounting. Watermarks derive pool pressure state: ready, pressure, critical,
draining, failed, or closed. Pending reclaim bytes are not immediately free.

## AllocationRequest And AllocationLease

`AllocationRequest` describes logical requirements: bytes, alignment,
allocation class, memory domain, lifetime class, residency requirement,
mutability, and reclaimability. It is not a native allocator argument list.

`AllocationLease` binds successful logical allocation to pool-backed storage:
pool, block, offset, length, alignment, generation, state, and optional
completion token. It never exposes native pointer semantics.

## AllocationPlan

`AllocationPlan` describes how Prepared execution storage is satisfied from
logical pools. It contains stable identity, generation, scope, pool bindings,
slots, lifetime intervals, reuse groups, reservation requirements, and guards.
The identity includes graph fingerprint, workload envelope, workspace
fingerprint, memory-domain requirements, pool policy version, and allocation
policy version. It excludes native addresses.

Plans can be cached, but cached plans must be revalidated against current pool
availability, reservations, Provider and Device compatibility, workspace
requirements, and alignment. Compatible allocation strategy changes mark a plan
stale; hard incompatibility invalidates it for new work.

## Reuse And Fragmentation

Temporal storage reuse is distinct from semantic Tensor aliasing. Independent
Tensor Resources may reuse the same backing only when lifetime intervals do not
overlap and CompletionToken barriers prove prior Device use is complete.

Fragmentation is tracked separately from total capacity. A pool can have enough
aggregate free bytes while lacking a compatible contiguous region. Compaction
and relocation are explicit Runtime-controlled work and must reject pinned,
mapped, in-flight, or address-stable Resources.

## KV Pages And Batch Workspace

KV cache can use a logical `KVPagePool` with page size, total pages, free pages,
leased pages, and pending reclaim pages. Page size follows KV-cache format and
model execution requirements rather than native allocator buckets. Pages are
owned by Sessions, sequences, or Prefix Cache references and recycle only after
ownership ends, shared references are gone, and completion is safe.

Continuous batching should use reusable batch workspace slots while preserving
CompletionToken barriers and protected KV/weight reservations.

## OOM, Fallback, And Observability

OOM categories distinguish pool capacity, Device capacity, reservation
conflict, fragmentation, alignment, pinned capacity, KV page exhaustion,
workspace exhaustion, and Provider allocation failure. Retry is bounded.
Fallback can trim optional caches, drop optional replicas, choose a
lower-workspace Kernel, use an alternate Plan or Device, spill where policy
permits, or reject admission.

Observability reports logical pool IDs, classes, capacity summaries,
fragmentation, reclaim, KV page, plan-cache, and OOM/fallback events. It must
redact native handles, native addresses, Tensor contents, weights, KV contents,
prompts, secrets, and credentials.
