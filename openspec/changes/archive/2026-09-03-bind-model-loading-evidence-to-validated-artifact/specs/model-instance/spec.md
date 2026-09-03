## MODIFIED Requirements

### Requirement: Model Instance Readiness

Model Instance lifecycle and readiness SHALL be distinct.

Readiness SHALL consider residency, Provider readiness, Device readiness,
adapter state, memory pressure, Runtime policy, architecture implementation
readiness, and weight materialization state.

Readiness-relevant facts the Runtime can itself observe -- including whether
mandatory weight resources are bound, whether a pinned Provider actually
resolves and offers an execution API, and whether a pinned Device is
available -- SHALL be derived from actual Runtime state, not accepted
outright from a caller-supplied claim. A caller MAY assert a stricter
(`false`) value than Runtime state alone would produce; a caller SHALL NOT
be able to assert a Runtime-observable fact as `true` when the Runtime does
not itself observe it as true.

A bound weight resource SHALL only count toward `weights_materialized` if
the Model Instance holds Runtime-issued materialization evidence -- minted
only by the one authorized weight-materialization transaction on successful
commit, never settable by an external caller -- whose recorded
`ModelArtifactId` matches this instance's own declared artifact and whose
recorded resource-id set matches the instance's currently-bound weight
resources exactly. Materialization evidence minted for one Model Instance
SHALL NOT cause any other Model Instance -- whether bound to the same or a
different Model Artifact -- to be treated as materialized. When the loaded
artifact declares a mandatory tensor inventory, every one of those tensors
SHALL be bound; a partial subset, however individually evidenced, SHALL NOT
count as materialized. A pinned Provider SHALL only count toward
`provider_ready` if its own status model reports it as currently accepting
new work, not merely that it is registered and exposes an execution
interface in principle.

The public surface for producing a `Ready` Model Instance SHALL NOT permit
an external caller to reach `Ready` other than through a path that performs
this derivation; lifecycle and readiness state SHALL NOT be directly
settable by an external caller. Weight resource bindings themselves SHALL
NOT be directly settable by an external caller either -- the only way to
bind a weight resource is for the one authorized materialization
transaction to commit successfully.

Readiness derivation SHALL NOT depend on a Provider-specific storage
readback capability (such as a host-memory tensor readback API). A Provider
that never implements host-memory readback for its resident storage SHALL
be able to reach `weights_materialized: true` through the same
Runtime-issued evidence every other Provider uses.

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

#### Scenario: Caller cannot forge a Runtime-observable fact

Given a Model Instance has no weight resources bound

When a caller requests warmup asserting weights are materialized

Then the Runtime's own observation of empty resource bindings overrides the caller's claim and the instance does not become Ready.

#### Scenario: A bound weight without a residency record does not count

Given a Model Instance has a weight resource identifier bound with no corresponding residency record

When a caller requests warmup asserting weights are materialized

Then the Runtime does not treat that resource as materialized and the instance does not become Ready.

#### Scenario: An incomplete mandatory weight inventory does not count

Given the loaded artifact declares multiple mandatory weight tensors and only some are bound, each with real residency and Runtime-issued materialization evidence

When a caller requests warmup asserting weights are materialized

Then the Runtime does not treat the instance as fully materialized and it does not become Ready.

#### Scenario: A Provider that rejects new work does not count as ready

Given a Model Instance is pinned to a Provider that is registered and exposes an execution interface

But that Provider's own status model reports it does not currently accept new work

When a caller requests warmup asserting the Provider is ready

Then the Runtime does not treat the Provider as ready and the instance does not become Ready.

#### Scenario: A hand-written weight binding without matching materialization evidence does not count

Given a caller writes tensor bytes directly into Provider storage and records a matching residency and weight binding by hand, without going through the authorized materialization transaction

When a caller requests warmup asserting weights are materialized

Then the Runtime finds no matching materialization evidence for that binding and the instance does not become Ready.

#### Scenario: Materialization evidence from a different Model Instance does not count

Given Model Instance A has successfully materialized weights through the authorized transaction, and Model Instance B has not

When Model Instance B's weight resource bindings are set to match Model Instance A's, and a caller requests warmup for Model Instance B asserting weights are materialized

Then the Runtime finds no materialization evidence recorded for Model Instance B's own id and Model Instance B does not become Ready.

#### Scenario: Materialized weights for a mismatched artifact do not count

Given a Model Instance declares Model Artifact A but its recorded materialization evidence was minted while it declared a different Model Artifact

When a caller requests warmup asserting weights are materialized

Then the Runtime treats the evidence as not matching this instance's current artifact and the instance does not become Ready.

#### Scenario: A device-only Provider without host readback can still prove materialization

Given a Provider never implements host-memory tensor readback for its resident storage

When that Provider's weights are materialized through the authorized transaction and a caller requests warmup

Then the Runtime derives `weights_materialized: true` from materialization evidence alone, without calling any Provider-specific readback capability, and the instance becomes Ready.

## ADDED Requirements

### Requirement: Model Instance Weight Resource Bindings Are Runtime-Sealed

A Model Instance's weight resource bindings SHALL be mutable only by the Runtime's own authorized weight-materialization transaction.

The weight resource bindings (the mapping from declared tensor name to
bound `TensorResourceId`, and the memory allocations backing them) SHALL
NOT be insertable, removable, or wholly replaceable by an external caller
holding a mutable reference to a Model Instance -- including by direct field
access or by assigning another Model Instance's bindings onto it.

#### Scenario: Direct binding mutation is not possible through the public API

Given a caller holds a mutable reference to a Model Instance obtained through the Runtime's public API

When the caller attempts to insert a weight resource binding without invoking the authorized materialization transaction

Then no public API exists to perform that mutation directly, and the instance's weight bindings remain unchanged.
