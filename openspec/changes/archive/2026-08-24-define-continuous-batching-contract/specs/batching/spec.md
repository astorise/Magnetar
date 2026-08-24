## ADDED Requirements

### Requirement: Continuous Batching

Magnetar SHALL define Continuous Batching as Runtime/Scheduler orchestration for
multiple active or pending generation operations.

#### Scenario: Batch generation operations

Given multiple compatible generation operations are admitted

When Scheduler forms a batch

Then Runtime may execute them through continuous batching policy.

---

### Requirement: Batching Is Runtime-Owned

Runtime SHALL own batch identity, operation membership, slot assignment, scheduling, and cleanup.

Clients and Components SHALL NOT forge batch identity or slot assignment.

#### Scenario: Forged slot

Given a caller submits a fabricated batch slot

When Runtime validates it

Then Runtime rejects it as unauthorized or not found.

---

### Requirement: Batching Does Not Own Raw Memory

Continuous Batching SHALL NOT allocate raw memory directly.

Memory allocation SHALL go through Memory Manager.

#### Scenario: Batch needs logits buffer

Given a batch needs logits workspace

When resources are planned

Then Scheduler requests Memory Manager admission.

---

### Requirement: Batching Does Not Own Raw KV Cache

Continuous Batching SHALL reference KV cache through Runtime-managed cache
identities.

It SHALL NOT expose or mutate raw KV cache memory directly.

#### Scenario: Batch slot with KV cache

Given a batch slot uses KV cache

When decode runs

Then the slot references Runtime-owned KV cache state.

---

### Requirement: Operation Lifecycle

Batched operations SHALL have lifecycle state.

States SHOULD include admitted, queued, prefill-pending, prefilling,
decode-pending, decoding, streaming, completed, cancelled, failed, rejected, and
evicted.

#### Scenario: Operation completes

Given a batched operation reaches its finish reason

When output streaming completes

Then lifecycle becomes completed.

---

### Requirement: Prefill And Decode Are Scheduled Separately

Continuous Batching SHALL distinguish prefill scheduling from decode scheduling.

#### Scenario: Mixed workload

Given one operation needs prefill

And another needs decode

When Scheduler plans work

Then it may apply different policies for prefill and decode.

---

### Requirement: Batch Slots

Continuous Batching SHALL define Runtime-owned batch slots.

A slot MAY bind operation, session, model context, tokenizer, sequence length,
KV cache, Prefix Cache boundary, Provider/Device metadata, memory reservation,
priority, deadline, and cancellation state.

#### Scenario: Slot assigned

Given an operation is admitted

When Scheduler assigns it to a batch

Then Runtime creates or updates a slot with operation metadata.

---

### Requirement: Batch Compatibility

Only compatible operations SHALL share a batch execution step.

Compatibility SHOULD consider model context, architecture implementation,
compute dtype, tokenizer, Provider/Device placement, Resource Affinity, KV cache
layout, sequence constraints, sampling compatibility, and memory placement.

#### Scenario: Incompatible models

Given two operations use incompatible loaded model contexts

When Scheduler forms a batch

Then they are not placed in the same execution step.

---

### Requirement: Admission Before Batch Entry

Runtime SHALL perform admission before an operation enters a batch.

#### Scenario: Invalid request

Given a generation request has invalid sampling parameters

When submitted

Then Runtime rejects it before batch entry.

---

### Requirement: Scheduling Policy

Continuous Batching SHALL define scheduling policy.

Policy MAY include FIFO, priority, deadline, fairness, weighted fairness,
latency target, throughput target, decode priority, prefill priority,
starvation prevention, queue limits, and memory pressure adaptation.

#### Scenario: Queue timeout

Given an operation waits beyond max queue time

When Scheduler evaluates it

Then Runtime applies timeout or priority policy.

---

### Requirement: Fairness

Scheduler SHALL prevent starvation where policy requires.

#### Scenario: Old operation waiting

Given an operation has waited beyond fairness threshold

When Scheduler selects work

Then policy increases its chance of scheduling or reports starvation prevention.

---

### Requirement: Backpressure

Continuous Batching SHALL support backpressure from queues, Providers, Devices,
Memory Manager, streaming consumers, session limits, and Runtime shutdown.

#### Scenario: Provider saturated

Given Provider pressure is saturated

When new operations are admitted

Then Runtime queues, rejects, or delays them according to policy.

---

### Requirement: Streaming Order

Batched streaming SHALL preserve per-operation token order.

#### Scenario: Interleaved batch

Given operations A and B are decoded in interleaved steps

When tokens are streamed

Then A's tokens remain ordered within A, and B's tokens remain ordered within B.

---

### Requirement: Cancellation

Continuous Batching SHALL support cancellation of queued, prefilling, decoding,
streaming, session-level, and shutdown-level work.

#### Scenario: Cancel queued operation

Given an operation is queued

When cancellation is requested

Then Runtime removes or marks it cancelled according to policy.

---

### Requirement: Failure Isolation

Failure of one operation SHALL not automatically fail unrelated operations unless
continuation is impossible.

#### Scenario: One operation fails

Given operation A fails due to invalid state

When operation B is independent

Then operation B may continue.

---

### Requirement: Dynamic Batch Resizing

Continuous Batching SHALL preserve operation state correctness when batches are resized dynamically.

#### Scenario: Operation finishes

Given one operation finishes

When Scheduler forms the next decode batch

Then the finished operation is removed from active slots.

---

### Requirement: Paged Attention Ready

Continuous Batching SHALL support metadata needed for future paged attention and SHALL NOT assume KV cache is one contiguous allocation.

#### Scenario: Paged KV cache

Given KV cache metadata uses pages

When Scheduler assigns slots

Then it preserves page/block metadata through Runtime-managed references.

---

### Requirement: Browser-Compatible Batching

Continuous Batching SHALL be platform-neutral and SHALL not require Wasmtime or
native Provider loading.

#### Scenario: Browser target

Given Runtime runs on browser target

When continuous batching is unavailable

Then Runtime returns browser-feature-unsupported or disables batching according
to policy.

---

### Requirement: Batching Error Categories

Batching failures SHALL use structured error categories.

#### Scenario: Queue full

Given operation queue is full

When a new operation is submitted

Then Runtime returns queue-full or batch-admission-rejected.

---

### Requirement: Batching Observability

Runtime SHALL support observations for operation admission, rejection, queueing,
batch formation, resizing, scheduling, completion, cancellation, failure,
pressure, cache assignment, backpressure, fairness, and starvation prevention.

Observability SHALL not expose raw prompts, raw logits, raw KV cache contents,
or raw Provider handles by default.

#### Scenario: Batch formed

Given Scheduler forms a batch

When observability records it

Then Runtime emits redacted batch metadata.
