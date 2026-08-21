# Runtime Observability

Magnetar Runtime observations are stable, portable records of Runtime-owned
decisions. They describe what the Runtime decided and which stable Runtime
objects were involved. They do not expose Provider-native implementation
details such as raw pointers, native handles, backend queues, streams, storage
objects, allocator internals, credentials or ambient filesystem paths.

## Event Lifecycle

Runtime events are emitted for the major execution phases:

- Capability resolution
- Provider selection and rejected Providers
- Resource Affinity decisions
- transfer and materialization requirements
- Memory Planning
- Execution Planning
- Scheduling
- Provider submission
- execution start, completion, cancellation and interruption
- Provider and Device health changes
- structured diagnostics

Every event carries a `TraceId` and `SpanId`. Events may also carry a
`CorrelationId`, `ExecutionPlanId`, `ScheduledOperationId`, Provider, Device and
Capability binding. These identifiers are Runtime-owned and portable.

## Tracing

A `RuntimeTrace` groups related `RuntimeEvent` values by `TraceId`. Compute
Execution Plans create the trace used by later scheduling and Provider
execution events, so observers can reconstruct one execution without depending
on backend-specific identifiers.

## Metrics

Runtime metrics use stable kinds and explicit units. Initial metrics cover:

- queue latency
- planning latency
- execution latency
- memory usage estimates
- transfer volume
- materialization count
- Provider utilization

Metrics can reference the same trace, Provider and Device identifiers as events.

## Diagnostics

Runtime diagnostics use stable `RuntimeDiagnosticCode` values. Diagnostic
messages are redacted before publication when they contain native handles,
backend paths or other Provider-private details. Diagnostics reference Runtime
objects instead of native Provider objects.

## Exporter Components

Observability exporters are WASM Components that consume the stable
`magnetar:runtime/observability@1.0.0` input contract. Exporter Components may
transform observations to OpenTelemetry, Prometheus, Jaeger or custom sinks.

Exporter configuration and output sinks are outside Provider execution. Exporter
Components must not receive Provider-native handles, backend storage, queues,
streams, raw pointers, credentials or ambient filesystem paths.
