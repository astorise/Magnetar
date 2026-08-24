## ADDED Requirements

### Requirement: Generation Supports Batched Execution

Generation SHALL support Runtime/Scheduler-coordinated batched execution where
available.

#### Scenario: Batched decode

Given compatible operations are scheduled together

When decode runs

Then Generation semantics remain per operation.

---

### Requirement: Generation Stop Conditions Remain Per Operation

Stop conditions SHALL be evaluated per operation, even when execution is
batched.

#### Scenario: One operation stops

Given operation A reaches EOS

And operation B does not

When batched decode completes

Then operation A finishes and operation B may continue.

---

### Requirement: Generation Streaming Remains Per Operation

Streaming SHALL preserve per-operation identity and token order.

#### Scenario: Batched token output

Given a batch produces tokens for operations A and B

When Runtime streams output

Then each operation receives its own ordered token events.
