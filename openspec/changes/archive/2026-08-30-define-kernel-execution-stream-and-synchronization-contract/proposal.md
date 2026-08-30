# Define Kernel Execution Stream And Synchronization Contract

## Why

Magnetar now defines a Prepared Execution Plan capable of materializing:

- Kernel selection
- specialization
- Provider/Device placement
- memory planning
- Prepared Kernel bindings
- prepared execution segments

The remaining execution-path question is how those prepared operations are
submitted asynchronously while preserving dependency, Tensor Resource,
KV-cache, memory-reuse, cancellation, and completion semantics.

Native platforms expose very different synchronization primitives:

- CUDA streams and events
- HIP streams and events
- Metal command queues, command buffers, and shared events
- Vulkan queues, semaphores, fences, and timeline semaphores
- WebGPU queues
- OpenVINO asynchronous infer requests
- QNN execution objects
- CPU worker pools and task completion primitives

Magnetar SHALL NOT expose these native concepts through its Runtime contracts.

The Runtime must own logical execution dependencies.

The Provider must own native synchronization implementation.

The architectural invariant is:

```text
Runtime owns dependency semantics.
Provider owns native synchronization objects.
```

## What Changes

This change defines:

- ExecutionStream
- ExecutionStreamId
- ExecutionStreamClass
- CompletionToken
- ExecutionDependency
- ExecutionSubmission
- ResourceReadiness
- stream ordering
- cross-stream dependencies
- cross-Device dependencies
- cross-Provider dependencies
- asynchronous completion
- failure propagation
- Tensor Resource readiness
- memory-reuse fences
- KV-cache ordering
- continuous-batching synchronization
- cancellation semantics
- deadlines and waits
- Provider synchronization capability discovery
- optional Provider ABI synchronization extension
- Prepared Execution Plan stream bindings
- Provider prepared-segment completion
- observability
- conformance

## Core Model

The logical model is:

```text
PreparedExecutionPlan
        |
        +---- ExecutionStream compute-0
        |          |
        |          +-- Kernel A
        |          |      |
        |          |      +--> CompletionToken A
        |          |
        |          +-- Kernel B depends-on A
        |
        +---- ExecutionStream transfer-0
                   |
                   +-- Transfer C
                          |
                          +--> CompletionToken C
```

Cross-stream synchronization is explicit:

```text
compute-0:
    Kernel A
        |
        +---- CompletionToken A
                     |
                     v
transfer-0:
    Transfer B
        |
        +---- CompletionToken B
                     |
                     v
compute-1:
    Kernel C
```

No CUDA event, Vulkan semaphore, Metal event, native queue, or native stream is
exposed to Runtime callers.

## ExecutionStream

ExecutionStream is a Runtime logical ordered submission lane associated with a
Provider and compatible Device execution context.

Conceptually:

```text
ExecutionStream
    id
    class
    ProviderBinding
    DeviceBinding
    generation
    priority_hint
    state
```

ExecutionStream SHALL NOT contain a native queue/stream pointer.

## ExecutionStreamId

ExecutionStreamId SHALL be opaque.

A Provider or Runtime MAY internally use a numeric identifier.

A numeric representation SHALL NOT imply pointer semantics.

ExecutionStreamId SHALL NOT be portable across Runtime instances unless an
explicit reconstruction mechanism exists.

## Execution Stream Class

ExecutionStream SHOULD carry an extensible class identity.

Baseline semantic classes MAY include:

```text
magnetar:execution/compute
magnetar:execution/transfer
magnetar:execution/control
```

The class vocabulary SHALL remain extensible.

A class is a scheduling/execution hint.

It SHALL NOT expose native platform queue types.

## Logical Stream Versus Native Queue

One Runtime ExecutionStream does not require one native Provider stream.

A Provider MAY map:

```text
many logical streams -> one native queue
one logical stream   -> one native queue
one logical stream   -> native task pool
```

provided the advertised ordering and dependency semantics are preserved.

## Same-Stream Ordering

Submissions to the same ExecutionStream SHALL observe logical ordering unless
an explicitly advertised relaxed execution mode states otherwise.

The baseline contract SHOULD provide ordered semantics:

```text
submit A
submit B
```

means B SHALL not observe execution effects that violate A-before-B dependency
semantics for that stream.

Provider MAY execute internally in parallel only if observable contract
semantics remain equivalent.

## Cross-Stream Ordering

Different ExecutionStreams SHALL NOT imply ordering merely because submissions
were made sequentially by the host.

Ordering across streams SHALL require an explicit ExecutionDependency or
equivalent Plan dependency.

## CompletionToken

Every asynchronously submitted operation or prepared segment MAY produce a
CompletionToken.

Conceptually:

```text
CompletionToken
    id
    stream
    generation
    state
```

Suggested states:

```text
pending
completed
failed
cancelled
lost
```

CompletionToken is a Runtime-visible opaque synchronization fact.

It SHALL NOT expose the Provider's native event/fence object.

## CompletionToken Semantics

A completed token SHALL mean all execution effects covered by that submission
are complete and visible according to the Provider's execution contract.

For a Kernel producing Tensor Resources, completion means required output writes
are ready for dependent consumers.

## Completion Scope

A CompletionToken SHALL identify its logical completion scope.

The scope MAY represent:

- one Kernel submission
- one transfer
- one prepared segment
- one execution-plan stage
- a Provider batch submission

A larger scope SHALL not weaken correctness.

## ExecutionDependency

ExecutionDependency represents a logical ordering requirement.

It SHOULD reference one or more CompletionTokens or resource-readiness
requirements.

Conceptually:

```text
ExecutionDependency
    predecessors
    scope
```

## Dependency Satisfaction

A dependent operation SHALL not execute until required predecessor dependencies
are satisfied.

Provider MAY implement this using:

- native stream events
- timeline semaphores
- command-buffer dependencies
- native fences
- task graph dependencies
- host-mediated waits

The implementation remains Provider-private.

## Device-Side Dependency

Provider SHOULD use native Device-side synchronization where available and
policy permits it.

Runtime SHALL NOT require host synchronization simply because dependency exists.

## No Automatic Global Synchronization

Runtime SHALL avoid global Device synchronization as the normal mechanism for
dependency ordering.

Operations analogous to:

```text
cudaDeviceSynchronize()
```

SHOULD NOT be required between every Kernel.

Global synchronization MAY be used only where necessary or explicitly
requested by a diagnostic/management policy.

## Host Wait

Runtime MAY await a CompletionToken when host-visible completion is required.

Typical cases include:

- returning final inference output
- reading host-visible diagnostics
- shutting down execution state
- safe resource destruction
- explicit synchronization API

Host wait SHALL support structured failure.

## Host Readiness

A Tensor Resource SHALL NOT be read by the host before all required writes have
completed and host visibility requirements are satisfied.

## Non-Blocking Poll

Runtime SHOULD support non-blocking completion observation where practical.

Conceptually:

```text
pending
completed
failed
```

This permits Scheduler and Runtime to progress without blocking worker threads.

## ExecutionSubmission

A logical submission SHOULD identify:

- ExecutionStream
- PreparedKernelId or PreparedExecutionSegment
- input resources
- output resources
- workspace resources
- dependencies
- execution metadata
- optional deadline
- cancellation scope

Native pointer arguments SHALL remain inside Provider-private resource
resolution.

## PreparedExecutionPlan Integration

Prepared Execution Plan MAY precompute:

- stream assignments
- dependency edges
- resource readiness edges
- segment boundaries
- synchronization scopes

Example:

```text
PlanNodeBinding
    stream_class = compute
    dependencies = [node-17]
```

At execution time Runtime resolves logical Plan stream bindings into active
ExecutionStream instances.

## Static Dependency Plan

Dependencies derivable purely from Execution Graph semantics SHOULD be
precomputed during Plan construction.

The token hot path SHOULD not rediscover the complete dependency graph.

## Dynamic Dependencies

Some dependencies are invocation/session-specific.

Examples:

- current Session KV-cache write
- continuous-batch slot ownership
- asynchronous adapter readiness
- request cancellation

These MAY be bound dynamically while preserving the prepared dependency
structure.

## Prepared Segment Synchronization

A Provider-prepared execution segment MAY contain internal synchronization that
is opaque to Runtime.

From Runtime's perspective the segment SHALL expose:

- declared input readiness requirements
- declared output resources
- CompletionToken
- structured failure

Internal native event graphs remain Provider-owned.

## ResourceReadiness

Tensor Resources SHALL have logical readiness state.

Conceptually a write may establish:

```text
ResourceReadiness
    resource
    completion_token
    access_scope
```

Consumers SHALL depend on required readiness before accessing the resource.

## Last Writer Tracking

Runtime/Memory Manager SHOULD be able to identify the most recent pending writer
for a Tensor Resource or relevant resource region.

A consumer SHALL not read data before that writer completes unless the Provider
contract explicitly guarantees equivalent ordered execution.

## Read Dependencies

Multiple read-only consumers MAY execute concurrently when:

- all required previous writes are complete or properly dependency-ordered
- resource affinity permits access
- no conflicting writer exists

## Write Dependencies

A writer SHALL be ordered against conflicting previous reads/writes according to
resource aliasing and lifetime semantics.

## Resource Regions

Future implementations MAY track dependencies at sub-resource/region level.

The baseline contract MAY conservatively track a whole Tensor Resource.

Conservative synchronization is valid.

Unsafe missing synchronization is not.

## Aliasing

When two logical Tensor Resources alias the same underlying allocation,
Memory Manager/Runtime SHALL ensure dependencies account for overlapping
access.

A Plan SHALL NOT assume independence merely from distinct Resource IDs if
aliasing metadata states otherwise.

## Memory Reuse Fence

Memory Manager SHALL not reuse allocation storage while unfinished work may
still access it.

Conceptually:

```text
allocation
   |
   +-- active CompletionToken A
   +-- active CompletionToken B
```

The allocation becomes reusable only after all required lifetime dependencies
have completed.

## Resource Retirement

Destroying a Tensor Resource handle SHALL not immediately free/reuse native
storage if asynchronous execution still references it.

Logical destruction and physical reuse MAY therefore occur at different times.

## Workspace Reuse

Prepared Execution Plans MAY aggressively reuse workspace allocations.

Workspace reuse SHALL be synchronized against completion of the prior user.

## Transfer Operations

Data movement MAY execute asynchronously on transfer-capable streams.

A transfer SHALL produce normal CompletionToken semantics.

Example:

```text
GPU tensor
   |
transfer stream
   |
CompletionToken T
   |
CPU/other consumer
```

## Transfer And Compute Overlap

Provider MAY overlap compute and transfer when supported.

Runtime SHALL express ordering only for actual dependencies rather than forcing
unnecessary serialization.

## Resource Affinity

An ExecutionStream SHALL be bound to a compatible Provider/Device context.

A Tensor Resource incompatible with that context SHALL not be submitted
silently.

Required data movement remains explicit.

## Cross-Device Dependency

Dependencies between Devices MAY be represented logically.

Provider MAY satisfy them through:

- device-native peer synchronization
- shared synchronization primitives
- runtime-mediated completion
- explicit transfer

Runtime SHALL not assume peer synchronization exists.

## Cross-Provider Dependency

Native synchronization handles SHALL NOT be exchanged directly between
different Providers through Core contracts.

Cross-Provider dependencies SHALL be Runtime-mediated unless a future explicit
interoperability capability defines a safe neutral contract.

Baseline behavior may be:

```text
Provider A CompletionToken
        |
        v
Runtime observes completion
        |
        v
Provider B submission
```

This may be less efficient but remains portable and safe.

## Provider Interoperability

A future Provider interoperability capability MAY optimize cross-Provider
dependencies.

It SHALL NOT expose raw native events through public Magnetar contracts.

## KV Cache Ordering

KV-cache updates are execution dependencies.

For a Session/sequence:

```text
decode step N writes KV[N]
            |
            v
decode step N+1 reads KV[0..N]
```

Runtime SHALL ensure step N+1 cannot observe incomplete required KV writes.

## Independent Sequences

Independent continuous-batching sequences MAY progress concurrently where
their resource dependencies do not conflict.

KV ordering SHALL be sequence/resource-aware rather than forcing unnecessary
global synchronization.

## Paged KV Cache

Paged KV-cache page allocation, mutation, reuse, and retirement SHALL observe
completion dependencies.

A page SHALL NOT be reassigned while unfinished execution may still access it.

## Prefix Cache

If a Prefix Cache resource is shared read-only after construction, its
construction/write completion SHALL be established before consumers use it.

## Continuous Batching

Continuous batching SHALL integrate CompletionToken state into slot lifecycle.

A batch slot SHALL not be reused if previous asynchronous execution still
references its resources.

## Batch Step Completion

A continuous-batching execution step MAY produce one completion scope covering
multiple sequences.

Runtime SHALL retain enough association to update per-sequence state safely.

## Scheduler Boundary

Scheduler owns:

- admission
- batching
- ordering policy
- workload prioritization

Scheduler SHALL NOT own native stream/queue objects.

Runtime execution subsystem translates Scheduler decisions into logical
ExecutionStreams.

## Stream Priority

ExecutionStream MAY carry a priority hint.

Priority is advisory.

Provider SHALL map it according to capability.

No priority hint SHALL bypass:

- correctness
- dependency
- Resource Affinity
- admission
- memory safety

## Priority Vocabulary

Priority SHOULD use a small portable policy representation rather than
Provider-native integer queue priorities.

Example conceptual values MAY include:

```text
background
normal
latency-sensitive
```

The exact vocabulary SHOULD remain extensible.

## Fairness

Provider/Runtime MAY apply fairness across logical streams.

Priority SHALL not imply indefinite starvation of other admitted work unless an
explicit deployment policy allows it.

## Cancellation

Cancellation SHALL be logical first.

When an inference request is cancelled:

- new dependent work SHOULD stop being submitted
- not-yet-started work SHOULD be cancelled where possible
- in-flight Device work MAY continue if native cancellation is unavailable
- resources referenced by in-flight work SHALL remain alive
- cancelled outputs SHALL not be published as successful inference results

## Cancellation Capability

Provider SHALL advertise cancellation semantics.

Suggested capability levels:

```text
not-supported
before-submit-only
queued-work
cooperative
interruptible
provider-specific
```

## Cancellation Does Not Mean Completion

A cancellation request SHALL NOT imply that Device work has stopped.

Resource reuse SHALL wait for actual completion/quiescence.

## Cancellation Token State

If in-flight work completes after logical cancellation, its CompletionToken MAY
record completion while the owning inference operation remains cancelled.

Runtime SHALL distinguish:

```text
physical execution completed
```

from:

```text
request result accepted
```

## Deadlines

Execution submission MAY carry deadlines or timeout policy.

A deadline SHALL NOT permit unsafe destruction of in-flight resources.

On deadline expiration Runtime MAY:

- stop future submissions
- request Provider cancellation
- return deadline failure
- retain resources until quiescent

## Dependency Failure

If a predecessor CompletionToken fails, dependent work SHALL not execute unless
explicit recovery policy defines an alternate path.

Failure SHALL propagate structurally.

## Failure Propagation

A dependent operation SHOULD receive a failure identifying the predecessor or
dependency class that prevented execution.

## Provider Failure

Provider failure MAY transition outstanding CompletionTokens to:

```text
failed
lost
```

depending on whether completion can still be determined.

## Device Loss

Device loss SHALL invalidate affected ExecutionStreams and Prepared Execution
Plans for new work.

Outstanding resources SHALL follow Provider/Runtime failure recovery policy.

Runtime SHALL not assume unfinished writes became valid.

## Lost Completion

If Runtime can no longer determine whether native execution completed, affected
resources SHALL not be unsafely reused.

Policy MAY require:

- Provider reset
- Device reset
- resource invalidation
- Model Instance failure

## Stream Lifecycle

Suggested ExecutionStream states are:

```text
creating
ready
draining
failed
closed
```

## Stream Drain

Runtime SHOULD support draining a stream.

Draining means:

- no new normal work accepted
- existing submitted work allowed to complete
- stream closes after outstanding tokens become terminal

## Stream Destruction

A stream SHALL not destroy Provider-native synchronization state while
outstanding work still relies on it unless Provider explicitly guarantees safe
destruction semantics.

## CompletionToken Lifetime

CompletionToken SHALL remain resolvable until:

- all interested Runtime dependencies have consumed its terminal state
- dependent resource lifetime tracking no longer needs it
- Provider/native completion state may be safely released

## Token Reuse

If numeric token IDs are reused, generation information or equivalent ABA-safe
identity SHALL prevent old dependencies from being confused with new work.

## Concurrency Safety

ExecutionStream and CompletionToken lifecycle operations SHALL be safe under
concurrent Runtime scheduling.

A token SHALL reach a terminal state exactly once.

## Provider Capability Discovery

Provider SHOULD advertise asynchronous execution capabilities including:

- asynchronous submission
- ordered stream semantics
- maximum/recommended logical concurrency
- cross-stream dependency support
- Device-side dependency support
- host wait
- non-blocking poll
- transfer overlap
- priority support
- cancellation level
- deadline support
- prepared-segment completion
- multi-Device dependency capability

## Capability Absence

A Provider without asynchronous execution MAY remain conformant by implementing
a synchronous baseline.

For a synchronous Provider:

```text
submit()
    -> operation completes
    -> CompletionToken immediately completed
```

This is valid but may offer less concurrency.

## Reference CPU

Reference CPU Provider MAY implement ExecutionStreams using:

- immediate synchronous execution
- worker pool
- bounded task executor

It SHALL preserve the same logical dependency semantics.

## Device Boundary

Device remains descriptive/status-oriented.

Device SHALL NOT expose methods such as:

```text
device.create_cuda_stream()
device.wait_event()
device.record_fence()
```

Native synchronization remains a Provider responsibility.

## Provider ABI

If synchronization crosses the native Provider ABI, the ABI extension SHALL be:

- explicitly versioned
- C-compatible
- optional
- free of Rust trait-object ABI
- based on opaque integer/token identifiers
- explicit about buffer ownership
- explicit about terminal-state retrieval
- protected against unwinding across ABI

Existing Providers without the extension SHALL remain valid through synchronous
fallback if their conformance profile allows it.

## Provider ABI Completion Operations

A conceptual optional ABI MAY provide operations equivalent to:

```text
create_execution_stream
release_execution_stream

submit_prepared_kernel
submit_prepared_segment

poll_completion
wait_completion
release_completion

request_cancel
```

Exact function signatures are not defined by this change.

## ABI Native Handle Privacy

ABI stream/token identifiers SHALL remain Provider-owned opaque values.

Runtime SHALL not reinterpret them as:

- `CUstream`
- `cudaEvent_t`
- `MTLCommandQueue*`
- `VkQueue`
- `VkSemaphore`
- `HANDLE`
- function pointer

## WIT Boundary

ExecutionStream native synchronization SHALL NOT become Component authority.

WASM Components SHALL continue expressing portable computation/data movement
requirements.

Components SHALL NOT receive native streams, queues, events, semaphores, or
CompletionToken-based hardware control.

## Runtime Inference API

Normal inference callers MAY observe high-level:

- completion
- streaming output
- cancellation
- deadline

They SHALL NOT receive Provider-native synchronization handles.

## Hot Path

Prepared Plan execution SHOULD reduce synchronization work to bounded logical
operations:

```text
resolve Plan stream
bind resources
collect required dependencies
submit
record CompletionToken
update resource readiness
```

No compilation, qualification, global Kernel search, or autotuning SHALL occur
as a synchronization side effect.

## Observability

Execution synchronization observability MAY include:

```text
stream-created
stream-draining
stream-closed

submission-created
submission-started
submission-completed
submission-failed

dependency-wait
completion-pending
completion-completed
completion-failed

resource-readiness-updated
memory-reuse-delayed

cancellation-requested
cancellation-deferred
deadline-exceeded
```

Observability MAY report:

- logical ExecutionStreamId
- CompletionTokenId
- Provider/Device stable identity
- Plan generation
- KernelId
- dependency count
- queue delay
- execution duration
- cancellation class
- failure category

Observability SHALL NOT report:

- native stream pointers
- native event pointers
- native queue handles
- raw Tensor addresses
- model weights
- KV contents
- prompts
- secrets
- credentials

## Error Model

Structured errors SHOULD include:

```text
execution-stream-unavailable
execution-stream-create-failed
execution-stream-not-ready
execution-stream-draining
execution-stream-failed
execution-stream-closed
execution-stream-provider-mismatch
execution-stream-device-mismatch
execution-stream-priority-unsupported

execution-submission-invalid
execution-submission-failed
execution-submission-dependency-invalid
execution-submission-resource-incompatible

execution-dependency-failed
execution-dependency-cycle
execution-dependency-cross-provider-unsupported

execution-completion-invalid
execution-completion-failed
execution-completion-lost
execution-completion-timeout
execution-completion-already-released

execution-resource-not-ready
execution-resource-write-conflict
execution-resource-read-conflict
execution-resource-affinity-invalid

execution-cancellation-unsupported
execution-cancellation-too-late
execution-cancellation-failed

execution-deadline-exceeded

execution-provider-synchronization-failed
execution-device-lost

internal-execution-synchronization-error
```

## Conformance

Conformance SHALL validate:

- Runtime owns logical dependencies
- Provider owns native synchronization
- Device exposes no native stream API
- same-stream ordered semantics
- cross-stream execution requires explicit dependency
- CompletionToken is opaque
- native event/queue handles do not leak
- dependent Kernel cannot observe unfinished writes
- Tensor Resource host access waits for readiness
- Memory Manager does not reuse active asynchronous storage
- aliased resources synchronize safely
- transfer/compute overlap remains possible
- cross-Provider dependencies are Runtime-mediated
- KV-cache ordering is preserved
- paged-KV pages are not reused too early
- continuous-batch slots are not reused too early
- cancellation does not imply physical completion
- cancelled work retains resources until quiescent
- deadline failure does not permit unsafe memory reuse
- failed dependencies stop dependent execution
- Device loss does not mark unfinished outputs ready
- stream retirement waits for outstanding work
- synchronous Provider fallback remains possible
- Prepared Plan does not expose native synchronization state
- synchronization observability is redacted

## Non-Goals

This change does not:

- expose CUDA streams
- expose Vulkan queues/semaphores
- expose Metal command queues
- define one native queue model
- require asynchronous execution
- define distributed synchronization between hosts
- define RDMA
- define NCCL collectives
- define multi-node tensor parallelism
- give Scheduler native Device queue authority
- let Components control native synchronization
- perform Kernel compilation
- perform Kernel selection
- redefine Memory Manager ownership
- define exact Provider ABI function signatures

## Impact

Magnetar gains a portable asynchronous execution substrate:

```text
PreparedExecutionPlan
        |
        v
logical ExecutionStreams
        |
        v
explicit dependencies
        |
        v
Provider-native synchronization
        |
        v
Device execution
```

This preserves high-performance asynchronous execution while keeping platform
synchronization details outside Runtime public contracts.