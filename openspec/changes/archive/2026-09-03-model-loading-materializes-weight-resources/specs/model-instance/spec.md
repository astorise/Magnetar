## MODIFIED Requirements

### Requirement: Model Instance Readiness

Model Instance lifecycle and readiness SHALL be distinct.

Readiness SHALL consider residency, Provider readiness, Device readiness,
adapter state, memory pressure, Runtime policy, architecture implementation
readiness, and weight materialization state.

#### Scenario: Provider not ready

Given an instance lifecycle exists

But Provider is not ready

When Runtime checks readiness

Then the instance is not ready for generation.

#### Scenario: Weights not materialized

Given an instance lifecycle exists

But its declared weights were never successfully materialized into Tensor
Resources

When Runtime checks readiness

Then the instance is not ready for generation, distinguishable from every
other readiness factor being satisfied.
