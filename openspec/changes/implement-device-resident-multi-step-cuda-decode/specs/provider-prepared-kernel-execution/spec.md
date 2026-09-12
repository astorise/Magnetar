## ADDED Requirements

### Requirement: Tensor Resource Copy Without Host Materialization

`ProviderExecutionApi` SHALL expose a way to duplicate a Tensor Resource's current bytes under a fresh, caller-chosen identity without requiring host-visible bytes to exist at any point during the copy, for a Provider that holds the source resource device-resident. The copy SHALL replace (and release) whatever the destination identity previously held, using the same admission and replacement discipline the existing host-typed admitted write already provides.

#### Scenario: Device-resident Provider copies a resource without host materialization

Given a Provider holds a Tensor Resource device-resident

When it duplicates that resource to a new identity through this contract

Then the copy completes without producing or requiring host-visible bytes for either identity

#### Scenario: Copy replaces a previously-held allocation at the destination identity

Given a destination identity already holds an allocation from a prior copy

When a new copy targets that same identity

Then the prior allocation is released and replaced, not left to accumulate alongside the new one

#### Scenario: Reference CPU continues to satisfy the contract

Given Reference CPU, a host-visible Provider

When it implements this copy contract

Then it produces the same observable result as reading the source and writing the destination, without being required to expose that as two separate host-visible steps
