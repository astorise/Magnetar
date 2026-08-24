# scheduler Specification

## Purpose
TBD - created by archiving change define-continuous-batching-contract. Update Purpose after archive.
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

