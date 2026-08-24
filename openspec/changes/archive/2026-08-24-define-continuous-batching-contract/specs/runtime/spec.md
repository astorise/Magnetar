## ADDED Requirements

### Requirement: Runtime Owns Batch Admission

Runtime SHALL own admission of operations into continuous batching.

#### Scenario: Admission denied

Given an operation violates policy

When it is submitted

Then Runtime rejects it before Scheduler batch entry.

---

### Requirement: Runtime Prevents Forged Batch State

Runtime SHALL reject client- or Component-forged batch IDs, slot IDs, Resource
Affinity, Provider placement, or KV cache placement.

#### Scenario: Forged batch slot

Given a request claims a privileged batch slot

When Runtime validates it

Then the claim is rejected or ignored.

---

### Requirement: Runtime Coordinates Batching Subsystems

Runtime SHALL coordinate Scheduler, Generation, Sampling, Memory Manager,
KV Cache, Prefix Cache, Provider, Device, and Session policy for continuous
batching.

#### Scenario: Decode batch

Given a decode batch is scheduled

When Runtime executes it

Then all subsystem constraints are applied before Provider submission.

---

### Requirement: Runtime Observes Continuous Batching

Runtime SHALL support structured observations for batching without exposing raw
prompts, logits, cache contents, or native handles.

#### Scenario: Operation queued

Given an operation is queued

When telemetry is emitted

Then Runtime records redacted queue metadata.
