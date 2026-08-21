# Tasks

## Observability Architecture

- [x] Define Observability Component role
- [x] Define Stream Exporter role
- [x] Define Snapshot Exposer role
- [x] Define Custom Observer role
- [x] Document that Observability Components are not Providers
- [x] Document that exporters do not participate in compute resolution

## Observability Emit Capability

- [x] Define `magnetar:observability/emit`
- [x] Define custom metric record
- [x] Define custom log record
- [x] Define custom event record
- [x] Define metric kind
- [x] Define metric tags
- [x] Define namespace validation
- [x] Define Component identity attachment
- [x] Define structured emission errors

## Observability Reader Capability

- [x] Define `magnetar:observability/reader`
- [x] Define RuntimeMetricsSnapshot
- [x] Define SchedulerMetricsSnapshot
- [x] Define ProviderMetricsSnapshot
- [x] Define DeviceMetricsSnapshot
- [x] Define ObservabilityMetricsSnapshot
- [x] Define dropped observation counters
- [x] Define snapshot consistency semantics
- [x] Define reader authorization

## Observability Stream Capability

- [x] Define `magnetar:observability/stream`
- [x] Define ObservationStream resource
- [x] Define ObservationFilter
- [x] Define ObservationBatch
- [x] Define ObservationRecord
- [x] Define stream subscription
- [x] Define bounded pull semantics
- [x] Define end-of-stream semantics
- [x] Define stream interruption semantics
- [x] Define observation ordering semantics

## Observation Records

- [x] Define Runtime event variant
- [x] Define Runtime metric variant
- [x] Define Runtime trace variant
- [x] Define Runtime diagnostic variant
- [x] Define Provider health variant
- [x] Define Device health variant
- [x] Define Scheduler variant
- [x] Define execution lifecycle variant
- [x] Define data movement variant
- [x] Define memory planning variant

## Correlation

- [x] Define TraceId
- [x] Define SpanId
- [x] Define CorrelationId
- [x] Attach ScheduledOperationId when applicable
- [x] Attach ComputeExecutionPlanId when applicable
- [x] Attach ProviderId when applicable
- [x] Attach DeviceId when applicable
- [x] Attach ComponentId when applicable

## Non-Blocking Runtime Delivery

- [x] Define bounded internal Observation Bus
- [x] Define non-blocking Runtime publication
- [x] Define try-emit behavior
- [x] Define queue capacity
- [x] Define dropped observation behavior
- [x] Increment dropped observation counter
- [x] Ensure Runtime execution never awaits exporter processing

## Aggregation

- [x] Define Runtime metrics aggregation worker
- [x] Aggregate counters
- [x] Aggregate gauges
- [x] Aggregate latency metrics
- [x] Aggregate Scheduler metrics
- [x] Aggregate Provider metrics
- [x] Aggregate Device metrics
- [x] Maintain snapshot independently of exporters

## Stream Exporters

- [x] Define OpenTelemetry exporter Component example
- [x] Define Jaeger exporter Component example
- [x] Define JSONL exporter Component example
- [x] Define custom stream exporter Component
- [x] Define event filter behavior
- [x] Define batching behavior
- [x] Define exporter flush behavior where applicable

## Snapshot Exposers

- [x] Define Prometheus exposition Component example
- [x] Consume `observability/reader`
- [x] Map counters to Prometheus counters
- [x] Map gauges to Prometheus gauges
- [x] Define histogram representation placeholder
- [x] Prevent Prometheus-specific types from entering Runtime core

## Sink Capabilities

- [x] Define explicit outbound HTTP dependency model
- [x] Define explicit filesystem write dependency model
- [x] Define explicit secret read dependency model
- [x] Define stdout/log output dependency model
- [x] Prevent ambient network access
- [x] Prevent ambient filesystem access
- [x] Prevent ambient secret access

## Component Scoping

- [x] Define Capability-level import authorization
- [x] Define endpoint scoping
- [x] Define filesystem path scoping
- [x] Define secret namespace scoping
- [x] Define observation category scoping
- [x] Define custom metric namespace scoping
- [x] Omit unauthorized imports from linker when possible
- [x] Validate value-based scopes when necessary

## Exporter Lifecycle

- [x] Define discovered state
- [x] Define loaded state
- [x] Define initializing state
- [x] Define active state
- [x] Define degraded state
- [x] Define saturated state
- [x] Define failed state
- [x] Define disabled state
- [x] Define stopped state

## Failure Isolation

- [x] Isolate exporter traps
- [x] Isolate exporter timeouts
- [x] Isolate sink failures
- [x] Isolate invalid exporter output
- [x] Preserve Runtime execution state after exporter failure
- [x] Preserve Scheduled Operation state after exporter failure
- [x] Report exporter failures separately

## Backpressure

- [x] Define exporter consumption limits
- [x] Define observation stream batch limits
- [x] Define exporter-side buffer limits
- [x] Define drop policy
- [x] Define disable-exporter policy
- [x] Define degradation policy
- [x] Prohibit compute-path blocking by default

## Load Shedding

- [x] Define observability as lower priority than compute execution
- [x] Define exporter load shedding
- [x] Define snapshot request shedding
- [x] Define exporter throttling
- [x] Preserve Runtime control operations under observability pressure

## Hot Reload

- [x] Define ObservabilityPolicy
- [x] Define hot-reloadable sampling rate
- [x] Define hot-reloadable log severity
- [x] Define hot-reloadable observation filters
- [x] Define hot-reloadable exporter state
- [x] Define hot-reloadable batching limits
- [x] Define hot-reloadable endpoints where safe

## Security and Privacy

- [x] Redact native Provider diagnostics
- [x] Redact credentials
- [x] Redact sensitive paths
- [x] Prevent native handle exposure
- [x] Prevent raw pointer exposure
- [x] Prevent backend storage exposure
- [x] Apply observation access policy

## Errors

- [x] Define invalid observation error
- [x] Define observation access denied error
- [x] Define observation stream closed error
- [x] Define exporter unavailable error
- [x] Define exporter saturated error
- [x] Define exporter failed error
- [x] Define sink unavailable error
- [x] Define sink unauthorized error
- [x] Define dropped observation diagnostic
- [x] Define invalid observability policy error

## Documentation

- [x] Document non-blocking observability architecture
- [x] Document stream versus snapshot model
- [x] Document OpenTelemetry exporter architecture
- [x] Document Prometheus exposition architecture
- [x] Document Jaeger exporter architecture
- [x] Document custom exporter architecture
- [x] Document Capability scoping
- [x] Document sink access
- [x] Document failure isolation
- [x] Document load shedding
- [x] Document hot reload