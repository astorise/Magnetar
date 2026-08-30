## ADDED Requirements

### Requirement: Provider Owns Native Synchronization

Provider SHALL own native streams, queues, events, semaphores, fences, and
equivalent synchronization objects.

#### Scenario: Metal Provider

Given Runtime creates logical compute stream

When Provider realizes it

Then MTLCommandQueue remains Provider-private.

### Requirement: Provider Advertises Async Capability

Provider SHALL advertise supported asynchronous execution features.

#### Scenario: GPU Provider discovery

Given Provider supports Device-side events and transfer overlap

When capability descriptor is read

Then these capabilities are explicitly discoverable.

### Requirement: Provider Preserves Logical Ordering

Provider SHALL preserve advertised ExecutionStream ordering semantics regardless
of internal implementation.

#### Scenario: Internal task pool

Given Provider maps one logical stream to multiple workers

When ordered submissions execute

Then observable dependencies remain equivalent to ordered stream contract.

### Requirement: Provider May Optimize Dependencies Device-Side

Provider SHALL be allowed to implement logical dependencies using native Device-side
synchronization.

#### Scenario: CUDA event

Given stream B depends on stream A

When CUDA Provider supports event wait

Then Provider may implement dependency without blocking host.

### Requirement: Provider Does Not Expose Native Event

Provider SHALL NOT return native synchronization pointer through Runtime public
contracts.

#### Scenario: Completion generated

Given native event exists

When Runtime receives completion identity

Then it receives opaque token only.

### Requirement: Provider Supports Structured Completion Failure

Provider SHALL report asynchronous execution failure through CompletionToken or
equivalent structured completion state.

#### Scenario: Kernel launch failure detected asynchronously

Given failure occurs after submission

When completion is polled

Then structured failed state is returned.

### Requirement: Provider Advertises Cancellation Semantics

Provider SHALL accurately describe cancellation capability.

#### Scenario: CUDA work already submitted

Given Provider cannot interrupt running Kernel

When cancellation capability is queried

Then it SHALL NOT advertise interruptible cancellation.

### Requirement: Provider Stream Failure Is Contained

Failure of one logical stream SHALL NOT silently mark unrelated completed
resources invalid unless Device/Provider failure scope requires it.

#### Scenario: One queued submission fails

Given Device remains healthy

When failure is reported

Then unrelated streams follow Provider's declared failure scope.

### Requirement: Provider Native Graph Completion Is Opaque

Provider-prepared execution segment SHALL expose only logical completion to
Runtime.

#### Scenario: OpenVINO async request

Given Provider uses native request object

When segment completes

Then native request object remains private.
