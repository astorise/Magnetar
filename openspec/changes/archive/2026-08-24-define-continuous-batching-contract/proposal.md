# Define Continuous Batching Contract

## Why

Magnetar now has contracts for:

- Model Artifacts
- Model Loading
- Tokenizer
- Generation
- Sampling
- Inference Sessions
- KV Cache
- Prefix Cache
- Memory Manager
- Providers and Devices

The next missing foundation is Continuous Batching.

Modern inference runtimes need to handle many concurrent generation requests
efficiently.

A single request may be in prefill.

Another may be in decode.

Another may be waiting for KV cache allocation.

Another may reuse a Prefix Cache entry.

Another may be cancelled.

Another may be streaming tokens.

Without a first-class Continuous Batching contract, batching behavior may become
hidden inside:

- Generation
- Scheduler
- Provider execution
- KV cache implementation
- Prefix Cache lookup
- Memory Manager admission
- Session state

That would make fairness, cancellation, memory pressure, batch slot ownership,
streaming order, latency policy, Provider pressure, and Resource Affinity unsafe
or inconsistent.

This change defines Continuous Batching as a Runtime/Scheduler orchestration
contract.

## What Changes

This change introduces Continuous Batching as a first-class inference scheduling
model.

Continuous Batching SHALL coordinate multiple generation operations over time.

It SHALL support:

- request admission
- prefill batching
- decode batching
- batch slots
- operation lifecycle
- token step scheduling
- KV cache assignment
- Prefix Cache reuse
- memory admission
- Provider/Device placement
- cancellation
- backpressure
- fairness policy
- priority policy
- latency policy
- streaming token ordering
- structured errors
- observability

Continuous Batching SHALL not own raw memory, raw KV cache content, Provider
handles, Device handles, or raw logits.

## Continuous Batching Is Runtime-Owned

Continuous Batching SHALL be owned by the Runtime and Scheduler.

Clients and Components SHALL NOT forge batch identity, batch slots, KV cache
slots, Resource Affinity, or Provider placement.

Batch identifiers SHALL be opaque Runtime-issued identifiers.

## Continuous Batch

A Continuous Batch represents a Runtime-managed set of active or pending
generation operations that may share scheduling cycles.

A batch MAY include:

- prefill work
- decode work
- active operation slots
- queued operation slots
- memory reservations
- KV cache references
- Prefix Cache lookup results
- Provider/Device placement metadata
- scheduling policy
- streaming output routing
- cancellation state

## Operation Lifecycle

A batched generation operation SHALL have lifecycle state.

Initial states SHOULD include:

```text
admitted
queued
prefill-pending
prefilling
decode-pending
decoding
streaming
completed
cancelled
failed
rejected
evicted
```

Semantics:

- `admitted`: Runtime accepted the operation
- `queued`: operation waits for scheduling
- `prefill-pending`: operation waits for prefill resources
- `prefilling`: prompt prefill is executing
- `decode-pending`: operation waits for decode step
- `decoding`: decode step is executing
- `streaming`: output token event is being delivered
- `completed`: generation finished
- `cancelled`: cancellation completed
- `failed`: operation failed
- `rejected`: operation was not admitted
- `evicted`: operation state was evicted according to policy

## Prefill Versus Decode Scheduling

Continuous Batching SHALL distinguish prefill scheduling from decode scheduling.

Prefill may process many prompt tokens and require large compute and memory.

Decode usually advances active sequences one or more tokens at a time.

The Scheduler SHALL be able to apply different policies to prefill and decode.

Examples:

```text
prefill-heavy batch
decode-heavy batch
mixed batch
prefill-first policy
decode-priority policy
latency-priority policy
throughput-priority policy
```

## Batch Slots

Continuous Batching SHALL define batch slots.

A batch slot represents an execution position for an active operation in a
batched step.

A slot MAY be associated with:

- operation ID
- session ID
- model context
- tokenizer reference
- current sequence length
- generated token count
- KV cache reference
- Prefix Cache reuse boundary
- Provider/Device placement
- memory reservation
- priority
- deadline
- cancellation state

Slot identity SHALL be Runtime-owned and opaque.

## Batch Slot Assignment

Slot assignment SHALL be controlled by Runtime policy.

Assignment SHALL consider:

- model context compatibility
- tokenizer compatibility
- Provider compatibility
- Device compatibility
- Resource Affinity
- KV cache residency
- Prefix Cache reuse
- sequence length
- memory budget
- Provider pressure
- Device pressure
- priority
- fairness
- latency targets
- cancellation state

## Batch Compatibility

Only compatible operations may share a batch execution step.

Compatibility SHOULD consider:

- loaded model context
- architecture implementation
- compute dtype
- tokenizer compatibility
- Provider/Device placement
- Resource Affinity
- KV cache layout
- sequence length constraints
- sampling compatibility where Provider-assisted
- memory placement
- Runtime policy

A compatible batch does not require identical generation parameters unless the
execution path requires it.

Sampling may remain per-operation.

## Memory Manager Relationship

Continuous Batching SHALL use Memory Manager for admission and memory budgets.

Memory needs MAY include:

- batch input buffers
- batch output token buffers
- logits buffers
- attention masks
- position buffers
- sampling workspace
- KV cache blocks
- Prefix Cache lookup workspace
- temporary staging
- Provider-specific workspace

The Scheduler SHALL NOT allocate memory directly.

## KV Cache Relationship

Continuous Batching SHALL coordinate with KV Cache.

It may assign, reuse, grow, seal, evict, or release KV cache through
Runtime-managed cache APIs.

Batching SHALL NOT own raw KV cache memory.

Batch slots may reference KV cache resources.

KV cache Resource Affinity SHALL constrain scheduling.

## Prefix Cache Relationship

Continuous Batching MAY use Prefix Cache lookup before prefill.

Prefix Cache hits may reduce prefill work.

Partial prefix hits may change the prefill boundary.

Prefix Cache policy SHALL remain Runtime-owned.

Batching SHALL not bypass prefix privacy or sharing policy.

## Generation Relationship

Generation owns the generation operation semantics.

Continuous Batching coordinates when prefill and decode steps run.

Generation remains responsible for:

- request validation
- prefill/decode semantics
- stop conditions
- token streaming semantics
- state updates
- sampling invocation
- finish reasons

Batching SHALL not redefine generation behavior.

## Sampling Relationship

Sampling owns next-token selection.

Continuous Batching may group logits computation, but selection may remain
per-operation.

If Provider-assisted sampling is batched, Runtime SHALL validate compatibility
and preserve per-operation policy.

## Session Relationship

Batched operations may belong to Sessions.

Session policy SHALL constrain batching behavior.

Session policy may define:

- concurrency
- maximum active operations
- queueing allowed
- cancellation
- priority
- memory budget
- KV cache budget
- prefix cache use
- streaming allowed
- timeout

A session ID SHALL not grant batch authority.

## Provider Relationship

Continuous Batching SHALL use Runtime Resolution and Provider advertisements.

Provider compatibility MAY include:

- batch execution support
- max batch size
- max sequence length
- max total tokens
- supported dtypes
- supported layouts
- supported KV cache layout
- supported paged attention
- supported Provider-assisted sampling
- memory pressure
- admission state
- readiness

Provider pressure SHALL influence batch admission and size.

## Device Relationship

Device constraints SHALL influence batch planning.

Constraints MAY include:

- memory capacity
- current memory pressure
- compute capability
- max batch dimensions
- supported dtypes
- Resource Affinity
- transfer bandwidth
- readiness
- thermal or throttling signals where available

## Scheduling Policy

Continuous Batching SHALL define scheduling policy.

Policy MAY include:

- FIFO
- priority
- deadline
- fairness
- weighted fairness
- latency target
- throughput target
- decode priority
- prefill priority
- starvation prevention
- max queue time
- max active operations
- max batch tokens
- max batch sequences
- memory pressure adaptation

Default policy SHOULD be safe and predictable.

## Fairness

The Scheduler SHALL prevent indefinite starvation where policy requires.

Fairness may be defined across:

- sessions
- clients
- priorities
- operation age
- model contexts
- batch classes

## Backpressure

Continuous Batching SHALL support backpressure.

Backpressure may occur due to:

- queue length
- Provider pressure
- Device pressure
- memory pressure
- streaming consumer slowness
- session concurrency limit
- Runtime shutdown
- cancellation

Backpressure SHALL produce structured status or error.

## Streaming Ordering

Streaming output SHALL preserve per-operation token order.

Batch execution may interleave operations, but each operation's emitted tokens
SHALL remain ordered.

Streaming backpressure for one operation SHALL not corrupt another operation's
stream.

Policy determines whether one slow consumer can block, buffer, or cancel its
operation.

## Cancellation

Continuous Batching SHALL support cancellation.

Cancellation may target:

- queued operation
- active prefill
- active decode
- streaming operation
- entire session
- entire batch
- Runtime shutdown

Cancellation SHALL coordinate with Generation, KV Cache, Memory Manager,
Provider execution, and streaming output.

## Failure Isolation

Failure of one operation in a batch SHALL not automatically fail unrelated
operations unless the Provider, Device, Runtime, or batch step failure makes
continuation impossible.

Failures SHALL be mapped per operation where possible.

## Admission

Runtime SHALL perform admission before batching.

Admission SHOULD validate:

- model context availability
- tokenizer compatibility
- generation parameters
- sampling parameters
- session policy
- memory budget
- Provider readiness
- Device readiness
- queue capacity
- cancellation state

Rejected operations SHALL not enter the batch.

## Dynamic Batch Resizing

Continuous Batching MAY resize batches dynamically.

Resizing may occur due to:

- new operations arriving
- operations completing
- cancellation
- stop conditions
- memory pressure
- Provider pressure
- Device pressure
- streaming backpressure
- prefix cache hits
- KV cache growth

Resizing SHALL preserve operation state correctness.

## Paged Attention Readiness

Continuous Batching SHOULD support metadata needed for future paged attention.

It does not require immediate paged attention implementation.

Batching SHALL not assume KV cache is one contiguous allocation.

## Browser Target

Continuous Batching model SHALL be platform-neutral.

Browser targets may support reduced batching depending on:

- browser memory limits
- WebAssembly linear memory
- WebGPU capability
- available Provider model
- session policy

Unsupported batching features SHALL return structured errors.

Browser batching SHALL not require Wasmtime or native Provider loading.

## Error Model

Continuous Batching errors SHALL be structured.

Error categories SHOULD include:

- batch unavailable
- batch admission rejected
- queue full
- operation not found
- operation cancelled
- operation timed out
- session concurrency limit
- model incompatible
- tokenizer incompatible
- Provider unavailable
- Provider not ready
- Provider saturated
- Device unavailable
- Device memory insufficient
- memory admission failed
- Resource Affinity conflict
- KV cache unavailable
- KV cache incompatible
- Prefix Cache reuse denied
- batch compatibility failed
- batch size unsupported
- sequence length unsupported
- streaming backpressure
- scheduling policy denied
- runtime shutdown
- browser feature unsupported
- internal batching error

## Observability

Runtime SHOULD emit observations for:

- operation admitted
- operation rejected
- operation queued
- batch formed
- batch resized
- prefill scheduled
- decode scheduled
- batch submitted
- batch completed
- operation completed
- operation cancelled
- operation failed
- queue pressure
- memory pressure
- Provider pressure
- Device pressure
- prefix cache hit in batch
- KV cache assigned
- streaming backpressure
- fairness adjustment
- starvation prevented

Observability SHALL not log raw prompts, raw logits, raw KV cache contents, or
raw Provider handles by default.

## Non-Goals

This change does not:

- require a specific scheduler algorithm
- implement paged attention
- implement speculative decoding
- implement beam search
- define distributed batching
- define cross-node routing
- define Tachyon scheduling
- define remote serving protocol
- define client agent orchestration
- define filesystem/network/tool authority
- expose raw batch memory
- expose raw KV cache
- expose Provider handles
- require GPU hardware
- require browser batching implementation

## Impact

Magnetar gains a stable batching orchestration contract.

The inference runtime becomes able to reason about many active requests without
hiding batching inside Provider or Generation.

Conceptual execution becomes:

```text
Generation requests
        |
        v
Runtime admission
        |
        v
Continuous Scheduler
        |
        +-- prefill batches
        +-- decode batches
        +-- KV cache references
        +-- Prefix Cache hits
        +-- streaming outputs
        |
        v
Provider execution
```