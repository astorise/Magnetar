# Refine Provider Health, Readiness, and Pressure Model

## Why

Magnetar Providers are trusted native execution extensions.

They implement Capabilities and expose Devices.

The Runtime uses Providers for local inference execution through Capability
Resolution, Resource Affinity, execution planning, scheduling, and Provider
execution APIs.

The current Provider health model needs refinement before Provider ABI and
conformance are stabilized.

A Provider being loaded or reachable is not the same as being ready to accept
new inference work.

A Provider may be:

- alive but still initializing
- alive but overloaded
- alive but draining
- alive but only partially ready
- alive but unhealthy for one Device
- alive but unavailable for one Capability
- available for existing pinned resources but not for new work
- capable of completing in-flight operations but refusing admission
- stale because its last health report expired
- interrupted by hardware or driver conditions
- degraded but still usable when policy permits

Magnetar needs distinct concepts for:

```text
health
readiness
pressure
admission
drainage
staleness
device-level state
capability-level state
```

Without this separation, the Runtime may schedule work to a Provider that is
alive but saturated, or reject a Provider that is degraded but still valid for
specific pinned resources.

This change defines the refined Provider status model.

## What Changes

This change separates Provider status into multiple dimensions.

At minimum, Provider status SHALL distinguish:

- lifecycle state
- health state
- readiness state
- pressure state
- admission state
- drainage state
- report freshness
- Device-specific state
- Capability-specific state

These dimensions SHALL be represented in Runtime-owned status snapshots.

The exact Rust type names are implementation-defined.

### Provider Lifecycle

Provider lifecycle describes where a Provider is in its Runtime-managed life.

Lifecycle states SHOULD include:

```text
registered
loading
initializing
ready
draining
stopped
failed
removed
```

Lifecycle is about Runtime management.

It is not enough to decide whether a Provider should receive new work.

### Health

Health describes whether the Provider appears internally functional.

Health states SHOULD include:

```text
unknown
healthy
degraded
unhealthy
failed
```

Semantics:

- `unknown`: no reliable health signal exists
- `healthy`: no known internal fault
- `degraded`: Provider has a fault or limitation but may still serve some work
- `unhealthy`: Provider should not receive normal work
- `failed`: Provider is unusable until reinitialized or replaced

Health may be reported at several scopes:

- Provider
- Device
- Capability implementation

### Readiness

Readiness describes whether a Provider should receive new work.

Readiness states SHOULD include:

```text
not-ready
ready
read-only
draining
```

Semantics:

- `not-ready`: do not admit new work
- `ready`: may admit compatible new work
- `read-only`: may serve operations that do not create new mutable execution
  state, if applicable
- `draining`: may complete existing work but should not receive new work

A Provider can be healthy but not ready.

For example, a Provider may be:

```text
health = healthy
readiness = not-ready
reason = warming model cache
```

### Pressure

Pressure describes current load and capacity.

Pressure SHALL be separate from health.

Pressure levels SHOULD include:

```text
unknown
low
moderate
high
saturated
```

Pressure MAY be computed from signals such as:

- active operation count
- queued operation count
- estimated queue delay
- memory pressure
- device memory pressure
- allocator fragmentation
- compute utilization
- stream occupancy
- batch capacity
- admission tokens
- throttling state

Pressure may influence Resolution Policy and Scheduler admission.

Pressure alone does not necessarily mean failure.

### Admission

Admission describes whether the Provider currently accepts new work for a
specific scope.

Admission MAY be represented as:

```text
admit
prefer-not
reject
```

Admission MAY be scoped by:

- Provider
- Device
- Capability
- operation family
- memory requirement
- Resource Affinity
- existing pinned state

Admission SHALL be derived from readiness, pressure, policy, and compatibility.

### Drainage

A Provider may be placed in draining state.

A draining Provider SHALL:

- stop accepting ordinary new work
- continue existing operations where safe
- preserve pinned resources according to Resource Affinity
- allow explicit migration/materialization only when supported and authorized
- report when it has no remaining in-flight work
- transition to stopped or removed according to Runtime policy

Drainage SHALL NOT silently migrate Provider-owned resources.

### Staleness and TTL

Provider status reports SHALL have timestamps or monotonic freshness metadata.

Runtime SHALL treat stale Provider status as unreliable.

Every status signal that affects scheduling SHOULD have a TTL.

If a status report expires, the Runtime SHALL degrade its confidence.

A stale Provider SHALL not be treated as fully ready unless policy explicitly
allows it.

### Device-Level Status

Provider-level health is not sufficient.

Each Device exposed by a Provider MAY have distinct:

- health
- readiness
- pressure
- memory pressure
- availability
- interruption state

A Provider may be healthy while one Device is unavailable.

### Capability-Level Status

A Provider may implement several Capabilities.

Each Capability implementation MAY have distinct readiness.

For example, a Provider may be ready for:

```text
magnetar:compute/run
```

but not ready for a future:

```text
magnetar:generation/session
```

Capability-level status SHALL be considered during Capability Resolution.

### Operation-Family Status

Where supported, a Provider MAY report readiness or pressure by operation
family.

For example:

- matmul saturated
- memory transfer unavailable
- dtype conversion unavailable
- graph execution ready
- generation session creation blocked

Operation-family status is optional.

If present, Resolution and Planning MAY use it.

### Existing Pinned Resources

Provider readiness for new work SHALL be separated from validity of existing
Provider-owned resources.

A Provider may reject new unpinned work but still be required for operations on
existing Provider-pinned resources.

Resource Affinity remains authoritative.

Readiness SHALL NOT allow Resolution Policy to ignore affinity.

### Interruption State

Provider status SHALL represent interruption-related conditions where relevant.

Examples include:

- GPU reset
- driver loss
- device removed
- process interruption
- allocator failure
- OOM recovery
- cancellation storm
- thermal throttling
- maintenance drain

Such state may affect health, readiness, pressure, and admission.

### Scheduler Interaction

The Scheduler SHALL use resolved plans.

It SHALL not independently invent Provider selection.

However, Scheduler admission may reject or delay execution if the selected
Provider or Device becomes not-ready, saturated, draining, or stale before
submission.

If rejection occurs, Runtime policy decides whether to:

- retry resolution
- fail the operation
- queue until readiness returns
- require explicit migration
- cancel

### Resolution Interaction

Resolution SHALL consider Provider and Device status.

At minimum, Resolution SHALL:

- filter failed Providers
- filter incompatible unavailable Devices
- prefer ready Providers over not-ready Providers
- penalize degraded Providers according to policy
- penalize high-pressure Providers according to policy
- reject saturated Providers unless policy allows queued admission
- preserve Resource Affinity constraints
- avoid draining Providers for new unpinned work
- allow pinned work on draining Providers only when policy permits

### Observability

Provider status changes SHALL be observable.

Observations MAY include:

- lifecycle transitions
- health changes
- readiness changes
- pressure changes
- stale report detection
- admission decisions
- drain start/end
- Device status changes
- Capability status changes
- rejected candidates
- saturation events
- recovery events

Observability SHALL not control status.

It reports decisions and signals.

### Diagnostics

Resolution and execution diagnostics SHOULD explain why a Provider was:

- selected
- penalized
- skipped
- rejected
- drained
- marked stale
- marked unavailable

Diagnostics SHALL be stable and redacted.

They SHALL not expose unsafe native handles.

### Provider Reporting API

The native Provider interface SHALL expose status in a way that the Runtime can
interpret consistently.

This change defines the semantic model.

The exact ABI is deferred to:

```text
define-provider-loading-and-abi-policy
```

A temporary in-process Rust API MAY expose the refined status model before ABI
stabilization.

### Policy

Runtime policy SHALL determine how to treat degraded or pressured Providers.

Examples:

- allow degraded Provider for low-priority work
- reject degraded Provider for critical work
- prefer low pressure
- allow queueing on high pressure
- reject saturated Provider
- allow pinned work during drain
- fail stale status closed
- allow stale status for development mode

Policy decisions SHALL be explicit.

## Non-Goals

This change does not:

- stabilize Provider ABI
- define dynamic library loading ABI
- implement complete Provider conformance suite
- implement distributed health
- implement cluster health
- define Tachyon health propagation
- define remote Provider execution
- define model residency
- define full inference scheduler
- define cross-node failover
- silently migrate pinned resources
- make observability authoritative
- replace Resource Affinity
- replace Resolution Policy

## Impact

Provider status becomes suitable for real inference scheduling.

The Runtime can distinguish:

```text
Provider alive
Provider healthy
Provider ready
Provider under pressure
Provider accepting admission
Provider draining
Provider stale
```

Resolution and Scheduling become safer.

Provider behavior becomes easier to test.

Future Provider ABI work can encode a stable status surface rather than
guessing what `health` means.