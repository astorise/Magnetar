# execution-stream Specification

## Purpose
TBD - created by archiving change define-kernel-execution-stream-and-synchronization-contract. Update Purpose after archive.
## Requirements
### Requirement: Logical Execution Stream

Runtime SHALL expose a logical ExecutionStream abstraction for ordered Provider
submission without exposing native queue or stream objects.

#### Scenario: CUDA Provider

Given Runtime creates a compute ExecutionStream

When CUDA Provider realizes it

Then Provider may use a CUDA stream internally but Runtime receives only an
opaque logical identifier.

### Requirement: Stream Is Provider And Device Bound

ExecutionStream SHALL be associated with a compatible Provider and Device
context.

#### Scenario: Wrong Device resource

Given ExecutionStream targets Device A

And Tensor Resource is exclusively affine to Device B

When Runtime attempts submission without explicit movement

Then submission fails compatibility validation.

### Requirement: Same-Stream Ordering

Baseline ExecutionStream SHALL preserve ordered submission semantics.

#### Scenario: A then B

Given Kernel A is submitted before Kernel B on same stream

When B consumes A output

Then Provider preserves required A-before-B ordering.

### Requirement: Cross-Stream Ordering Is Explicit

Different ExecutionStreams SHALL not imply execution ordering without explicit
dependency.

#### Scenario: Host submits stream A then stream B

Given no dependency exists

When Provider executes

Then Runtime cannot rely on host submission order for synchronization.

### Requirement: Completion Token

Asynchronous submission SHALL produce an opaque CompletionToken or equivalent
immediately-terminal token for synchronous Providers.

#### Scenario: GPU Kernel submitted

Given execution continues asynchronously

When submit returns

Then Runtime receives CompletionToken without native event handle.

### Requirement: Completion Token Is Opaque

Runtime SHALL NOT reinterpret CompletionToken identity as a native
synchronization handle.

#### Scenario: Numeric token

Given token identifier is 4242

When Runtime stores it

Then 4242 has no pointer, event, semaphore, or queue semantics.

### Requirement: Explicit Dependency

Runtime SHALL be able to bind dependent submission to predecessor
CompletionToken.

#### Scenario: Cross-stream transfer

Given compute stream produces Tensor

And transfer stream consumes Tensor

When transfer is submitted

Then compute completion is an explicit dependency.

### Requirement: Provider Native Synchronization Is Private

Provider SHALL retain native event/fence/semaphore state behind the logical
dependency contract.

#### Scenario: Vulkan Provider

Given dependency uses a timeline semaphore

When Runtime diagnostics inspect dependency

Then VkSemaphore is absent.

### Requirement: No Mandatory Global Synchronization

Runtime SHALL not require Device-global synchronization for ordinary dependency
ordering.

#### Scenario: Independent Kernels

Given Kernels execute on independent streams

When no dependency exists

Then Runtime allows concurrency rather than globally synchronizing Device.

### Requirement: Non-Blocking Completion Query

Runtime SHALL support observing completion without blocking.

#### Scenario: Work still running

Given CompletionToken is pending

When Runtime polls it

Then pending is returned without waiting for Device completion.

### Requirement: Explicit Host Wait

Runtime SHALL support explicit waiting for CompletionToken when host-visible completion is
required.

#### Scenario: Final logits copied to host

Given transfer is asynchronous

When Runtime must return host-visible result

Then it waits for required completion before reading output.

### Requirement: Cancellation Does Not Imply Physical Completion

Logical cancellation SHALL not make in-flight resources immediately reusable.

#### Scenario: GPU operation cannot be interrupted

Given request is cancelled

And Provider reports cancellation unsupported for in-flight work

When Runtime handles cancellation

Then resources remain retained until CompletionToken reaches terminal physical
state.

### Requirement: Dependency Failure Propagates

Failed predecessor SHALL prevent ordinary dependent execution.

#### Scenario: Kernel A fails

Given Kernel B depends on A

When A CompletionToken fails

Then B is not executed as if A succeeded.

### Requirement: Stream Drain

Runtime SHALL support draining ExecutionStream before destruction.

#### Scenario: Runtime shutdown

Given three submissions remain pending

When stream enters draining

Then no new normal work is accepted and outstanding submissions may complete.

### Requirement: Synchronous Provider Compatibility

Provider SHALL be allowed to implement synchronous execution using the same
logical contract.

#### Scenario: Reference CPU immediate execution

Given CPU Kernel finishes before submit returns

When CompletionToken is returned

Then it is already completed.

### Requirement: Data Movement Uses Execution Synchronization

Asynchronous residency-changing movement SHALL integrate with ExecutionStream
and CompletionToken semantics.

#### Scenario: Host to GPU upload

Given transfer occurs asynchronously

When movement is submitted

Then destination readiness depends on transfer CompletionToken.

### Requirement: Source Remains Alive During Transfer

Source Resource SHALL not be destroyed/reused before asynchronous movement
finishes.

#### Scenario: Upload input buffer

Given host input upload remains pending

When logical input owner releases it

Then physical source storage remains alive until transfer completion.

### Requirement: Destination Is Not Ready Before Transfer Completion

Destination Resource SHALL not be consumed before transfer completion or
equivalent dependency ordering.

#### Scenario: MatMul after upload

Given GPU upload pending

When MatMul is submitted on another stream

Then explicit dependency orders MatMul after transfer.

### Requirement: Mapping Conflicts Respect Execution Dependencies

Host mapping SHALL synchronize with incompatible Device accesses.

#### Scenario: Host write during GPU read

Given GPU Kernel reads Resource

When host requests write mapping

Then mapping waits or fails according to policy.

### Requirement: Zero Copy Still Uses Readiness

Directly accessible shared Resource SHALL still follow ResourceReadiness.

#### Scenario: Shared host/GPU memory

Given GPU writes shared allocation

When CPU reads it zero-copy

Then host visibility/completion is established first.

### Requirement: Allocation Reuse Is Completion Token Aware

Execution completion SHALL govern when pool-backed slots SHALL be reused.

#### Scenario: Workspace reused across decode steps

Given step N workspace is still referenced by pending Kernel

When step N+1 begins

Then same slot is reused only after ordering guarantees safety.

### Requirement: Deferred Reclaim Is Observable To Allocator

A logically released AllocationLease SHALL remain pending until associated
CompletionTokens terminate.

#### Scenario: Cancelled request

Given Device work continues

When request resources are released logically

Then allocator does not treat storage as free yet.

### Requirement: Transfer Staging Pool Lifetime Is Synchronized

Transfer-staging allocation SHALL remain valid until asynchronous transfer
completes.

#### Scenario: Host-to-Device transfer

Given staging lease is used by transfer stream

When caller releases input

Then staging storage is reclaimed only after transfer completion.

