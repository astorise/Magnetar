## ADDED Requirements
### Requirement: Cross Device Stage Dependency Is Explicit

Downstream Device stage SHALL depend on upstream computation and required
movement completion.

#### Scenario: GPU0 to GPU1 pipeline

Given GPU0 produces activation

When GPU1 stage begins

Then upstream CompletionToken and transfer readiness are satisfied.

### Requirement: Peer Transfer Uses Logical Execution Stream

Direct Device-to-Device transfer SHALL be able to execute asynchronously through a logical
transfer ExecutionStream.

#### Scenario: GPU peer copy

Given peer-copy capability exists

When activation moves

Then Runtime obtains logical CompletionToken without native event exposure.

### Requirement: Cross Device Resource Lifetime Is Preserved

Source and destination resources SHALL remain valid during asynchronous
inter-Device movement.

#### Scenario: Source stage retires

Given transfer still pending

When source temporary lifetime ends logically

Then physical storage remains until transfer no longer references it.

### Requirement: Device Failure Propagates To Dependent Completion

Loss of Device involved in pending execution/transfer SHALL fail or lose
affected logical completions.

#### Scenario: GPU0 resets during peer copy

Given GPU1 stage depends on transfer

When completion fails/lost

Then GPU1 stage does not consume destination as successfully ready.

### Requirement: Multi Device Execution Does Not Expose Native Streams

Logical cross-Device execution SHALL remain expressed through
ExecutionStream/CompletionToken contracts.

#### Scenario: CUDA peer transfer

Given two CUDA streams/events are used internally

When Runtime observes operation

Then native synchronization handles are absent.
