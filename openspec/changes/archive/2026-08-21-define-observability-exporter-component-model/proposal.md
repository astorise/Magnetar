# Define Observability Exporter Component Model

## Why

Magnetar now has a Runtime Observability Model capable of describing Runtime
activity using stable portable observations.

The Runtime can produce observations for:

- Capability Resolution
- Provider selection
- Resource Affinity
- Memory Planning
- Execution Planning
- Scheduling
- Provider execution
- Provider Health
- Device Health
- Data Movement
- Runtime errors

These observations must be exportable to external observability systems.

However, the Magnetar Runtime core must not depend directly on telemetry
backends such as:

- OpenTelemetry
- Prometheus
- Jaeger
- Loki
- Elastic
- proprietary enterprise collectors

Telemetry integrations must remain optional, replaceable and independently
deployable.

Magnetar SHALL therefore support observability integrations implemented as WASM
Components.

The architecture SHALL distinguish between two observability consumption models:

1. stream-based observation consumers
2. snapshot-based observation consumers

Stream-based consumers are appropriate for:

- traces
- events
- logs
- diagnostics
- OpenTelemetry
- Jaeger
- JSONL export
- custom event sinks

Snapshot-based consumers are appropriate for:

- counters
- gauges
- Runtime state snapshots
- Prometheus-style scrape endpoints
- autoscaling metrics
- administrative status views

The Runtime SHALL remain the authoritative producer and aggregator of
observability information.

Observability Components SHALL transform or expose Runtime observations but
SHALL NOT own authoritative Runtime execution state.

Observability must also remain outside the critical compute path.

A slow or failed exporter SHALL NOT delay or fail:

- Compute Graph execution
- Provider execution
- Scheduler progress
- Resource lifecycle
- Runtime control operations

The Runtime SHALL use bounded, non-blocking delivery between execution paths and
the observability subsystem.

When observability capacity is exhausted, observations MAY be dropped according
to policy rather than blocking compute execution.

## What Changes

This proposal introduces the Observability Exporter Component Model.

The model defines three portable observability surfaces:

- `magnetar:observability/emit`
- `magnetar:observability/reader`
- `magnetar:observability/stream`

### Observability Emit

`magnetar:observability/emit` allows Components to submit custom portable
observations to the Runtime observability plane.

It may support:

- custom metrics
- structured logs
- structured events
- Component diagnostics

The Runtime validates, scopes and aggregates these observations.

### Observability Reader

`magnetar:observability/reader` exposes aggregated Runtime observability state.

It is intended primarily for privileged system Components.

It may expose:

- counters
- gauges
- Scheduler state
- Provider state
- Device state
- queue depth
- memory pressure
- dropped observation counts
- aggregated execution metrics

Prometheus-style Components SHOULD consume this snapshot model rather than
requiring a continuous event stream.

### Observability Stream

`magnetar:observability/stream` exposes a typed stream of Runtime observations to
authorized Components.

It may expose:

- Runtime Events
- Runtime Traces
- Runtime Diagnostics
- Provider Health transitions
- Device Health transitions
- Scheduler transitions
- execution lifecycle observations

OpenTelemetry, Jaeger, JSONL and similar exporters SHOULD consume this model.

### WASM Observability Components

Observability integrations MAY be implemented as WASM Components.

Examples include:

- OpenTelemetry exporter Component
- Prometheus exposition Component
- Jaeger exporter Component
- JSONL exporter Component
- Loki exporter Component
- custom enterprise exporter Component

These Components SHALL consume stable Magnetar observability contracts.

They SHALL NOT receive Provider-native implementation objects.

### Sink Capabilities

Exporter Components SHALL NOT receive ambient network, filesystem or secret
access.

External sinks SHALL be accessed through explicitly granted Capabilities.

Examples include:

- outbound HTTP
- filesystem write
- secret read
- log output
- future message queue Capabilities

### Non-Blocking Runtime Delivery

The Runtime SHALL place observation delivery outside the compute critical path.

Runtime execution paths SHALL use bounded non-blocking publication where
possible.

Observability saturation SHALL NOT block compute execution.

Dropped observations SHALL be counted and observable.

### Failure Isolation

Exporter failures SHALL be isolated from Runtime execution.

Exporter failure SHALL NOT alter:

- Compute results
- Scheduled Operation state
- Provider execution state
- Tensor Resources
- Resource Affinity
- Memory Plans

### Policy and Scoping

Observability Components SHALL operate according to explicit Runtime policy.

Policy may restrict:

- imported Capabilities
- allowed endpoints
- readable observation categories
- writable metric namespaces
- secrets
- filesystem paths
- sampling
- batching
- buffer capacity
- exporter activation

Unauthorized imports SHOULD be omitted from the Component linker when possible.

### Hot Reload

Observability configuration SHOULD be updateable without restarting the Runtime.

Hot-reloadable configuration MAY include:

- exporter enabled state
- observation filters
- log levels
- sampling rates
- batch limits
- buffer limits
- drop policy
- endpoint configuration

## Impact

The Runtime remains independent of specific observability backends.

OpenTelemetry, Prometheus, Jaeger and custom integrations can evolve
independently as WASM Components.

The compute critical path remains isolated from telemetry latency and failures.

Prometheus can efficiently expose aggregate Runtime state through a snapshot
model.

Trace-oriented exporters can consume typed event streams without requiring
internal JSON or HTTP serialization between the Runtime and Component.

Access to external systems remains explicitly scoped through Capabilities.

The architecture supports future third-party observability Components without
expanding the Runtime core.