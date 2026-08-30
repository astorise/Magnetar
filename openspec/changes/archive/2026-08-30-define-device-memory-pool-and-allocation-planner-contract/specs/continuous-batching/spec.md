## ADDED Requirements
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