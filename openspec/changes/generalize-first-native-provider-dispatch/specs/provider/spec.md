## ADDED Requirements

### Requirement: Provider TensorValue Mutation Error Channel

The Provider Execution API's `TensorValue` mutation operations SHALL report
Provider-native write failures through a structured Provider execution error,
consistent with the equivalent `HostTensor`-typed write operation.

A `TensorValue` admission operation SHALL distinguish a Memory Manager
admission failure from a Provider-native write failure; it SHALL NOT report
both failure kinds through the same error type.

#### Scenario: TensorValue write fails after successful admission

Given a `TensorValue` admission operation successfully admits an allocation

When the subsequent Provider-native write of that value fails

Then the operation releases the allocation it admitted

And reports a structured Provider execution error distinguishable from a
Memory Manager admission failure.

#### Scenario: TensorValue admission fails before any write is attempted

Given a `TensorValue` admission operation cannot admit the required allocation

When admission fails

Then the operation reports a Memory Manager admission failure

And does not attempt the Provider-native write.

#### Scenario: Default TensorValue write succeeds

Given a Provider does not override the default `TensorValue` write behavior

When a caller writes a `TensorValue`

Then the operation reports success unless the Provider's own write
implementation fails.
