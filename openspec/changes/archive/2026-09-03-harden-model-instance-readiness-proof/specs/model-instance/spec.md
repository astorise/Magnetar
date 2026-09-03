## ADDED Requirements

### Requirement: Model Instance Resume Revalidates Readiness

Resuming a suspended Model Instance SHALL NOT transition it directly to
`Ready`. State that made the instance eligible for suspension (Provider
health, weight materialization evidence, Device availability) MAY have
changed while it was suspended; resume SHALL re-derive readiness against
current Runtime state through the same evidence-deriving path used to
reach `Ready` from any other lifecycle state, rather than assuming prior
readiness still holds.

#### Scenario: Resume rejects stale evidence

Given a Model Instance was Ready and is then suspended

But readiness-relevant Runtime state changes to invalidate that instance's prior evidence while it is suspended

When resume is requested

Then the instance does not become Ready and the resume request fails

#### Scenario: Resume succeeds when evidence still holds

Given a Model Instance was Ready and is then suspended

And all readiness-relevant Runtime state remains valid while suspended

When resume is requested

Then the instance becomes Ready again

## MODIFIED Requirements

### Requirement: Model Instance Readiness

Model Instance lifecycle and readiness SHALL be distinct.

Readiness SHALL consider residency, Provider readiness, Device readiness,
adapter state, memory pressure, Runtime policy, and architecture implementation
readiness.

Readiness-relevant facts the Runtime can itself observe -- including whether
mandatory weight resources are bound, whether a pinned Provider actually
resolves and offers an execution API, and whether a pinned Device is
available -- SHALL be derived from actual Runtime state, not accepted
outright from a caller-supplied claim. A caller MAY assert a stricter
(`false`) value than Runtime state alone would produce; a caller SHALL NOT
be able to assert a Runtime-observable fact as `true` when the Runtime does
not itself observe it as true.

A bound weight resource SHALL only count toward `weights_materialized` if
it has a corresponding residency record the Runtime itself recorded, and
that residency's recorded Provider SHALL itself currently hold the tensor
-- a residency record alone, without confirmation from the Provider it
claims, SHALL NOT count. When the loaded artifact declares a mandatory
tensor inventory, every one of those tensors SHALL be bound; a partial
subset, however individually well-evidenced, SHALL NOT count as
materialized. A pinned Provider SHALL only count toward `provider_ready`
if its own status model reports it as currently accepting new work, not
merely that it is registered and exposes an execution interface in
principle.

The public surface for producing a `Ready` Model Instance SHALL NOT permit
an external caller to reach `Ready` other than through a path that
performs this derivation; lifecycle and readiness state SHALL NOT be
directly settable by an external caller.

#### Scenario: Provider not ready

Given an instance lifecycle exists

But Provider is not ready

When Runtime checks readiness

Then the instance is not ready for generation.

#### Scenario: Caller cannot forge a Runtime-observable fact

Given a Model Instance has no weight resources bound

When a caller requests warmup asserting weights are materialized

Then the Runtime's own observation of empty resource bindings overrides the caller's claim and the instance does not become Ready.

#### Scenario: A bound weight without a residency record does not count

Given a Model Instance has a weight resource identifier bound with no corresponding residency record

When a caller requests warmup asserting weights are materialized

Then the Runtime does not treat that resource as materialized and the instance does not become Ready.

#### Scenario: A residency record without a real Provider write does not count

Given a Model Instance has a weight resource with a recorded residency, but the residency's claimed Provider never received a write for that resource

When a caller requests warmup asserting weights are materialized

Then the Runtime does not treat that resource as materialized and the instance does not become Ready.

#### Scenario: An incomplete mandatory weight inventory does not count

Given the loaded artifact declares multiple mandatory weight tensors and only some are bound, each with real residency and Provider-backed evidence

When a caller requests warmup asserting weights are materialized

Then the Runtime does not treat the instance as fully materialized and it does not become Ready.

#### Scenario: A Provider that rejects new work does not count as ready

Given a Model Instance is pinned to a Provider that is registered and exposes an execution interface

But that Provider's own status model reports it does not currently accept new work

When a caller requests warmup asserting the Provider is ready

Then the Runtime does not treat the Provider as ready and the instance does not become Ready.
