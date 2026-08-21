## MODIFIED Requirements

### Requirement: Provider Fallback

The runtime SHALL support fallback Providers while resolving work that has not
created or consumed Provider-owned state.

Once live resource affinity binds a call to a Provider or Device, the runtime
SHALL only use a Provider that satisfies the complete affinity constraint set.
An unavailable bound Provider SHALL produce a structured affinity failure
instead of implicit migration.

#### Scenario: Primary Provider unavailable before state creation

- **GIVEN** the preferred Provider cannot execute a Capability
- **AND** no live resource affinity constrains the call
- **AND** another compatible Provider exists
- **WHEN** execution is resolved
- **THEN** the runtime selects the fallback Provider

#### Scenario: Bound Provider unavailable after state creation

- **GIVEN** a live opaque resource bound to a Provider
- **WHEN** that Provider becomes unavailable for a dependent call
- **THEN** the runtime reports an affinity failure
- **AND** it does not select another Provider for that live resource
