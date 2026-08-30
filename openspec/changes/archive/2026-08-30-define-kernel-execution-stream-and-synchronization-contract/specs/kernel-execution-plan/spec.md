## ADDED Requirements

### Requirement: Plan May Precompute Stream Assignment

Prepared Execution Plan SHALL be able to bind nodes or segments to logical execution stream
classes.

#### Scenario: Attention decode

Given Plan uses compute and transfer operations

When Plan is built

Then stream assignments may be determined before token execution.

### Requirement: Plan May Precompute Dependency Edges

Static execution dependencies SHALL be materializable in Prepared Execution Plan.

#### Scenario: MatMul feeds RMSNorm

Given graph topology establishes dependency

When Plan is prepared

Then dependency does not need full rediscovery on every execution.

### Requirement: Plan Does Not Store Native Streams

Prepared Execution Plan SHALL not contain Provider-native synchronization
objects.

#### Scenario: CUDA stream exists

Given Provider created native stream

When Plan is serialized or inspected

Then only logical ExecutionStream binding is present.

### Requirement: Plan Supports Dynamic Dependency Slots

Prepared Plan SHALL be able to expose dynamic dependency slots for
Session/invocation state.

#### Scenario: Prior KV update

Given decode Plan is reused across Sessions

When Session executes next step

Then its current KV CompletionToken can be bound dynamically.

### Requirement: Prepared Segment Has Logical Completion

Provider-prepared segment SHALL expose logical CompletionToken after
submission.

#### Scenario: CUDA Graph launch

Given Provider launches captured graph

When submitted

Then Runtime receives one logical completion scope without observing internal
CUDA events.

### Requirement: Plan Generation Retirement Includes Stream Work

Prepared Plan SHALL remain alive while submissions using its stream/segment
bindings remain in-flight.

#### Scenario: Plan replacement

Given generation 10 is retiring

And one submission remains pending

When generation 11 becomes active

Then generation 10 resources are not destroyed prematurely.
