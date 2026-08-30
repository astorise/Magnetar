# continuous-batching Specification

## Purpose
TBD - created by archiving change define-kernel-execution-stream-and-synchronization-contract. Update Purpose after archive.
## Requirements
### Requirement: Continuous Batch Step Tracks Completion

Each asynchronous batch execution step SHALL expose completion sufficient to
protect batch and sequence resources.

#### Scenario: Batch of eight sequences

Given one Kernel invocation processes all sequences

When submitted

Then Runtime tracks completion before reusing affected batch resources.

### Requirement: Batch Slot Reuse Is Completion-Aware

Batch slot SHALL not be reassigned while previous in-flight execution still
references its resources.

#### Scenario: Sequence finishes logically

Given its slot's Kernel is still running

When new sequence arrives

Then slot is not reused prematurely.

### Requirement: Independent Work May Overlap

Continuous batching SHALL permit use of multiple ExecutionStreams where
dependency/resource semantics permit it.

#### Scenario: Prefill and decode workloads

Given resources and policy allow concurrency

When Scheduler admits both

Then Runtime may overlap execution without global Device synchronization.

### Requirement: Cancellation Stops Future Batch Work

Cancelled sequence SHALL be excluded from future submissions while preserving
resources needed by already submitted work.

#### Scenario: Sequence cancelled after batch submission

Given current batch is in-flight

When cancellation arrives

Then no future batch includes sequence, but current resources remain valid until
completion.

### Requirement: Batch Completion Attribution Is Explicit

When one CompletionToken represents several sequences, Runtime SHALL retain
enough logical mapping to safely update their state.

#### Scenario: Shared Attention Kernel completes

Given four sequences share one invocation

When token completes

Then Runtime can mark required per-sequence KV/resource transitions safely.

### Requirement: Continuous Batching Reuses Pool-Backed Slots

Continuous batching SHALL reuse planned batch/intermediate workspace rather
than allocating native memory for every scheduling quantum.

#### Scenario: Stable batch size range

Given Prepared Plan supports batch 1..16

When successive batches execute

Then compatible planned slots SHALL be reused.

### Requirement: Batch Slot Lease Is Completion Safe

Batch workspace SHALL not recycle while previous asynchronous execution
references it.

#### Scenario: Next scheduler quantum arrives early

Given prior batch remains in-flight

When next batch needs same slot

Then scheduler/runtime waits, uses another slot, or applies backpressure.

### Requirement: Batch Memory Demand Participates In Admission

Scheduler admission SHALL account for available compatible pool capacity.

#### Scenario: Large prefill batch

Given workspace pool cannot satisfy projected demand

When Scheduler evaluates batch

Then it SHALL split/defer batch rather than trigger uncontrolled OOM.

### Requirement: Batch Workspace Does Not Consume Protected KV Capacity Implicitly

Pool class isolation SHALL be preserved.

#### Scenario: Large prefill

Given KV hard reservation exists

When transient batch workspace grows

Then it cannot consume protected KV bytes without explicit policy.

