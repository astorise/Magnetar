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
it has a corresponding residency record the Runtime itself recorded; a
resource identifier present without one SHALL NOT count. A pinned
Provider SHALL only count toward `provider_ready` if its own status model
reports it as currently accepting new work, not merely that it is
registered and exposes an execution interface in principle.

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

#### Scenario: A Provider that rejects new work does not count as ready

Given a Model Instance is pinned to a Provider that is registered and exposes an execution interface

But that Provider's own status model reports it does not currently accept new work

When a caller requests warmup asserting the Provider is ready

Then the Runtime does not treat the Provider as ready and the instance does not become Ready.

---

### Requirement: Model Instance Warmup

Model Instance warmup MAY be supported and SHALL be policy-controlled.

Warmup failure SHALL prevent ready state.

Regardless of warmup policy, `readiness` SHALL NOT report `Ready` while
`lifecycle` has not itself reached a state that supports inference use. A
warmup policy that does not perform lifecycle transitions SHALL NOT be able
to publish `Ready` readiness as a side effect.

The primitives capable of transitioning a Model Instance to `Ready`
(the underlying lifecycle transition and the raw ready-marking operation)
SHALL NOT be reachable by a caller outside the Runtime's own
implementation. An external caller SHALL only be able to request warmup
through the Runtime-owned entry point that performs readiness derivation
first.

#### Scenario: Warmup failure

Given Provider warmup fails

When Runtime evaluates instance readiness

Then the instance becomes failed or not-ready according to policy.

#### Scenario: Disabled policy cannot forge readiness

Given a Model Instance is in a lifecycle state that does not support inference use

When warmup is invoked with a policy that does not transition the lifecycle

Then readiness does not report Ready even if the supplied checks would otherwise compute Ready

#### Scenario: The raw ready-marking primitive is not externally reachable

Given a caller external to the Runtime's own implementation holds a mutable reference to a Model Instance

When that caller attempts to invoke the underlying lifecycle transition or ready-marking operation directly, bypassing the Runtime-owned warmup entry point

Then no such path is available to that caller
