# Tasks

## Core Types

- [x] Define RuntimeEvent
- [x] Define RuntimeTrace
- [x] Define RuntimeMetric
- [x] Define RuntimeDiagnostic
- [x] Define CorrelationId
- [x] Define TraceId
- [x] Define SpanId

## Resolution

- [x] Emit Capability Resolution events
- [x] Emit Provider selection events
- [x] Emit rejected Provider events

## Affinity

- [x] Emit Resource Affinity decisions
- [x] Emit transfer requirements
- [x] Emit materialization requirements

## Planning

- [x] Emit Memory Planning events
- [x] Emit Execution Planning events
- [x] Emit Scheduling events

## Execution

- [x] Emit Provider submission
- [x] Emit execution start
- [x] Emit execution completion
- [x] Emit cancellation
- [x] Emit interruption

## Health

- [x] Emit Provider health changes
- [x] Emit Device health changes
- [x] Emit Scheduler backpressure

## Metrics

- [x] Queue latency
- [x] Planning latency
- [x] Execution latency
- [x] Memory usage estimates
- [x] Transfer volume
- [x] Materialization count

## Diagnostics

- [x] Stable diagnostic codes
- [x] Correlate diagnostics to Runtime objects
- [x] Redact Provider-private information

## Documentation

- [x] Document Runtime event lifecycle
- [x] Document tracing model
- [x] Document metrics model
- [x] Document diagnostic model

## Exporter Components

- [x] Define Observability Exporter Component role
- [x] Define exporter input event stream
- [x] Define exporter configuration boundary
- [x] Define exporter output sink boundary
- [x] Document OpenTelemetry exporter as Component
- [x] Document Prometheus exporter as Component
- [x] Document Jaeger exporter as Component
- [x] Document custom exporter support
