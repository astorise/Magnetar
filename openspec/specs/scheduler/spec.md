# scheduler Specification

## Purpose
This specification defines scheduler identity, queue policy, fairness, capacity, cancellation, batching interaction, and memory admission boundaries.
## Requirements
### Requirement: Scheduler Owns Continuous Batching Policy

Scheduler SHALL own continuous batching policy execution under Runtime control.

#### Scenario: Scheduler forms batch

Given admitted operations are waiting

When Scheduler evaluates policy

Then it forms compatible prefill or decode work.

---

### Requirement: Scheduler Does Not Allocate Memory Directly

Scheduler SHALL request Memory Manager admission and reservations rather than
allocating memory directly.

#### Scenario: Batch workspace needed

Given a decode batch requires workspace

When Scheduler plans it

Then it requests Memory Manager feasibility.

---

### Requirement: Scheduler Uses Provider Pressure

Scheduler SHALL consider Provider readiness, admission, and pressure when
forming batches.

#### Scenario: Provider saturated

Given Provider status reports saturated pressure

When Scheduler forms a batch

Then it reduces, delays, or rejects work according to policy.

---

### Requirement: Scheduler Preserves Resource Affinity

Scheduler SHALL preserve Resource Affinity from model residency, KV cache,
Prefix Cache, tensors, and Provider-owned resources.

#### Scenario: KV cache on Device A

Given a batch operation depends on KV cache on Device A

When Scheduler selects placement

Then it preserves compatible placement or requests explicit Runtime movement or
rebuild.

---

### Requirement: Scheduler Maintains Operation State

Scheduler SHALL maintain per-operation state independently within a batch.

#### Scenario: Operation B cancelled

Given operation B is cancelled

When operation A remains active

Then Scheduler does not corrupt operation A state.

### Requirement: Scheduler Uses Kernel Metadata

Scheduler SHALL use Kernel metadata for planning batched and asynchronous
execution.

#### Scenario: Batch size limit

Given a Kernel supports max batch size 8

When Scheduler forms a batch

Then it does not plan that Kernel for batch size 16.

---

### Requirement: Scheduler Respects Kernel Execution Mode

Scheduler SHALL respect Kernel execution mode, cancellation support, timeout,
and workspace lifetime.

#### Scenario: Asynchronous kernel

Given a Kernel is asynchronous

When Scheduler dispatches work

Then it tracks completion and resource lifetime accordingly.

---

### Requirement: Scheduler Does Not Select Raw Native Functions

Scheduler SHALL not select raw Kernel function pointers.

It SHALL operate on Runtime-validated Kernel metadata and invocations.

#### Scenario: Native function pointer

Given a Provider has internal native functions

When Scheduler plans execution

Then it uses Kernel metadata, not raw function addresses.

### Requirement: Scheduler Uses Kernel Registry Metadata

Scheduler SHALL be allowed to use Kernel Registry metadata for planning, batching, deadlines,
backpressure, and pressure-aware scheduling.

#### Scenario: Batch planning

Given Scheduler forms a batch

When it estimates feasibility

Then it may use Kernel metadata such as max batch size and workspace.

---

### Requirement: Scheduler Delegates Final Dispatch To Runtime

Scheduler SHALL delegate final Kernel Dispatch validation and invocation
creation to Runtime.

#### Scenario: Scheduled work ready

Given Scheduler selects work to run

When Kernel execution is needed

Then Runtime Kernel Dispatch performs final revalidation and Provider
invocation.

---

### Requirement: Scheduler Handles Dispatch Outcomes

Scheduler SHALL handle Kernel Dispatch outcomes such as completion, failure,
timeout, cancellation, fallback, and backpressure.

#### Scenario: Dispatch timeout

Given Kernel Dispatch times out

When Scheduler receives the result

Then Scheduler updates operation state according to policy.

---

### Requirement: Scheduler Supplies Workload Context

Runtime SHALL treat Scheduler-provided workload state as an optimization input rather than an eligibility constraint; Scheduler MAY provide such state.

#### Scenario: Continuous batch

Given batch contains 32 active sequences

When throughput selection occurs

Then batch width may influence ranking.

---

### Requirement: Scheduler Does Not Override Kernel Eligibility

Scheduler SHALL not force an ineligible Kernel for throughput.

#### Scenario: Batch favors GPU

Given GPU Kernel violates affinity

When Scheduler wants throughput

Then Runtime still excludes GPU Kernel.

---

### Requirement: Scheduler Does Not Own Kernel Policy

Scheduler MAY provide load/queue context, but Runtime selection policy SHALL
remain authoritative over the final Kernel decision.

#### Scenario: Queue pressure

Given Scheduler reports backlog

When Kernel choice changes

Then decision is made through Runtime policy.

### Requirement: Scheduler Does Not Own Native Streams

Scheduler SHALL operate on logical workload/admission concepts rather than
Provider-native stream or queue objects.

#### Scenario: CUDA scheduling

Given Scheduler chooses latency-sensitive decode work

When Runtime dispatches it

Then Scheduler does not receive CUstream.

### Requirement: Scheduler May Supply Concurrency Intent

Scheduler SHALL be allowed to supply logical concurrency/priority hints to
execution subsystem.

#### Scenario: Background prefill

Given decode is latency-sensitive

When Scheduler orders work

Then Runtime may map workloads to different logical ExecutionStreams according
to policy.

### Requirement: Scheduler Observes Completion Logically

Scheduler SHALL be allowed to use CompletionToken terminal state to advance batches and
Sessions.

#### Scenario: Decode batch completes

Given token transitions completed

When Scheduler processes event

Then next logical step may be admitted.

### Requirement: Scheduler Does Not Treat Cancellation As Completion

Scheduler SHALL distinguish cancelled request from physical execution
quiescence where resource lifetime depends on completion.

#### Scenario: Cancelled sequence slot

Given Device work remains pending

When Scheduler wants to reuse slot

Then Memory/Runtime readiness prevents unsafe reuse.

### Requirement: Scheduler May Respect Session Placement Affinity

Scheduler SHALL be able to use logical Session/Plan placement affinity when admitting work.

#### Scenario: Decode Session owns GPU1 KV

Given GPU1 Plan remains healthy

When next token is scheduled

Then Scheduler may prefer the same ready Plan.

### Requirement: Scheduler Does Not Perform Device Placement Optimization

Scheduler SHALL not independently override Runtime MultiDevicePlacementPlan.

#### Scenario: GPU0 queue shorter

Given active Plan requires GPU1 stage

When Scheduler sees queue pressure

Then it cannot silently move stage without valid replacement Plan.

### Requirement: Admission Is Per Placement Plan

Scheduler SHALL admit work only when required Plan Devices/resources are
available.

#### Scenario: Mandatory GPU1 unavailable

Given Plan needs GPU0 and GPU1

When new request arrives

Then Scheduler does not start only half the Plan.

### Requirement: Cross Device Backpressure Is Supported

A slow downstream Device SHALL be able to create backpressure on upstream stages.

#### Scenario: GPU1 stage saturated

Given GPU0 produces faster than GPU1 consumes

When queues reach policy limit

Then Scheduler may throttle upstream submissions.

