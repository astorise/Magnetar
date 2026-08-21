# Provider Health Model

Provider Health is the runtime model for observing Provider, Device and
Capability availability before resolution, planning, scheduling and Provider
execution submission.

Health is advisory and dynamic. It improves selection and admission decisions,
but it is not a success guarantee and it is not automatic failover. A Provider
that reports `available` may still fail during execution. A Provider that
becomes unavailable after resource creation does not allow the Runtime to
silently migrate Provider-pinned resources.

## States

The stable health states are:

- `unknown`: no fresh report is available.
- `initializing`: the Provider or Device is starting or discovering resources.
- `available`: work may be considered.
- `degraded`: work may be accepted with reduced reliability, capacity or
  performance.
- `saturated`: new admission should be delayed or rejected as backpressure.
- `draining`: existing work may finish, but new work should not be assigned.
- `unavailable`: new work must not be assigned.
- `interrupted`: running work cannot continue because of Provider, Device,
  resource or Runtime failure.

By default, only `available` and `degraded` accept new work. Policies may rank
or reject degraded candidates, but the state remains visible as stable
diagnostic metadata.

## Reports

`ProviderHealthReport`, `DeviceHealth` and `CapabilityHealth` use stable
identifiers and `HealthState`. Reports may include timestamps, time-to-live
metadata, redacted diagnostics, trace identifiers and advisory capacity hints.

Capacity hints include queue depth, available memory estimates, memory
pressure, active operation counts, maximum accepted operations and recommended
admission limits. They are policy inputs, not hard guarantees.

Diagnostics must not expose raw pointers, GPU pointers, queues, streams, locks,
file descriptors, backend storage, Provider handles, Device handles,
credentials or ambient filesystem paths.

## Runtime Flow

Resolution Policy receives Provider, Device and Capability health on each
`ResolutionCandidate`. Candidates with unavailable, interrupted, unknown,
initializing, saturated or draining health are rejected by default. The
`Availability` policy ranks healthier candidates before deterministic identity
ordering.

Execution Planning evaluates health while selecting the Provider and Device
for a new plan. The resulting plan records the selected Provider, Device,
Capability version, Resource Affinity and no-implicit-migration constraint.

The Scheduler rechecks Provider and Device health immediately before
submission. If selected health no longer accepts new work, the operation reaches
an interrupted or failed terminal state with a stable scheduler error such as
Provider unavailable, Device unavailable or Provider saturated.

The Provider Execution API maps health-related failures to stable runtime
errors. Backend-specific detail may be attached only as redacted diagnostics.

## No Automatic Failover

Health changes never override Resource Affinity. Provider-pinned resources
remain bound to the original Provider unless a future explicit transfer,
replay, reload or migration contract is present.

If running work loses its Provider or Device, the Runtime reports interruption
or failure. It does not silently choose another Provider, another Device, or an
unplanned data movement path.
