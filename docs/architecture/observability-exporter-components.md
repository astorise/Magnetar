# Observability Exporter Components

Magnetar observability integrations are WASM Components, not Providers. They
consume Runtime-owned observations and snapshots, transform or expose them, and
never participate in compute Provider resolution, Device selection, execution
planning, memory planning or Provider submission.

## Component Roles

`ObservabilityComponentDescriptor` models four roles:

- `ObservabilityComponent`: shared role for observability-only Components.
- `StreamExporter`: consumes `magnetar:observability/stream`.
- `SnapshotExposer`: consumes `magnetar:observability/reader`.
- `CustomObserver`: imports `magnetar:observability/emit`.

OpenTelemetry, Jaeger, JSONL and custom enterprise exporters are stream
exporters. Prometheus exposition is a snapshot exposer and maps Runtime
snapshots to Prometheus text outside Runtime core.

## Capability Surfaces

`magnetar:observability/emit` accepts scoped custom metrics, structured logs and
structured events from authorized Components. Metric namespaces are checked by
policy and rejected with structured errors when unauthorized.

`magnetar:observability/reader` exposes aggregated snapshots:

- `RuntimeMetricsSnapshot`
- `SchedulerMetricsSnapshot`
- `ProviderMetricsSnapshot`
- `DeviceMetricsSnapshot`
- `ObservabilityMetricsSnapshot`

Snapshots remain available independently of external exporters while the
observability subsystem is enabled.

`magnetar:observability/stream` exposes typed `ObservationRecord` batches
through bounded pull semantics. Filters select categories, severity, Provider,
Device, Component or subsystem. End-of-stream and interruption are explicit
batch or error states.

## Non-Blocking Delivery

`ObservationBus` is bounded and exposes `try_emit`. A full bus applies
`ObservationOverflowPolicy` instead of blocking compute execution by default.
Dropped observations increment a counter that is available through snapshots.
Exporter processing is lower priority than compute execution and Runtime control
operations unless a future policy explicitly opts into compute blocking.

## Sink Access And Scoping

Exporter Components do not receive ambient network, filesystem or secret access.
External dependencies are declared with `ObservabilitySinkDependency`:

- outbound HTTP endpoint scope
- filesystem write path scope
- secret namespace
- stdout/log output

Policy authorizes these scopes before the Runtime links or exposes the
corresponding Capability. Unauthorized imports can be omitted from the linker
when possible; value-based scopes are still validated by the sink Capability.

## Failure Isolation

Exporter traps, timeouts, sink failures, invalid output and saturation are
reported as `ObservabilityError` or `ExporterRuntimeStatus`. These states do not
change Compute results, Scheduled Operation terminal state, Provider execution
state, Tensor Resources, Resource Affinity or Memory Plans.

Lifecycle states are Runtime-visible:

- discovered
- loaded
- initializing
- active
- degraded
- saturated
- failed
- disabled
- stopped

## Security

Observation contracts use stable portable values. Runtime observations and
diagnostics must not expose Rust trait objects, callbacks, raw native handles,
raw pointers, GPU pointers, backend storage, native queues, native streams,
kernel objects, allocator internals, credentials or sensitive filesystem paths.

## Hot Reload

`ObservabilityPolicy` identifies hot-reloadable fields:

- sampling rate
- log severity
- observation filters
- exporter state
- batching limits
- endpoints where safe

Policy updates can disable exporters, adjust filters, change sampling and tune
batch sizes without restarting the Runtime.
