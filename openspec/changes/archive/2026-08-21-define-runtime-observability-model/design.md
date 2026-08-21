## Context

The Runtime already owns capability resolution, resource affinity, memory
planning, execution planning, scheduling, Provider submission and health
decisions. Existing code records stable diagnostics for many of these phases,
but there is no common observation contract that tools can consume across the
whole Runtime lifecycle.

Observability must be portable. It must describe Runtime decisions with stable
Runtime identifiers and must not expose Provider-native details such as raw
pointers, native handles, backend queues, streams, allocator internals, backend
storage, credentials or ambient filesystem paths.

## Goals / Non-Goals

**Goals:**

- Define portable Runtime observation types for events, traces, metrics and
  diagnostics.
- Correlate execution observations with `TraceId`, `SpanId`, optional
  `CorrelationId`, `ExecutionPlanId`, `ScheduledOperationId`, Provider, Device
  and Capability identifiers.
- Emit observations for resolution, Provider selection, rejected Providers,
  resource affinity, transfer and materialization requirements, memory planning,
  execution planning, scheduling, Provider submission, execution start,
  completion, cancellation, interruption and health changes.
- Keep diagnostics stable and redact Provider-private details before exposing
  them through Runtime observation APIs.
- Define exporter Components as consumers of the stable Runtime observability
  contract without coupling the Runtime to any one telemetry backend.

**Non-Goals:**

- Do not mandate OpenTelemetry, Prometheus, Jaeger or any other transport.
- Do not expose native Provider handles or backend implementation details.
- Do not make exporters part of Provider execution.
- Do not add external telemetry dependencies.

## Decisions

### Add a Runtime-owned observation model

Introduce `RuntimeEvent`, `RuntimeTrace`, `RuntimeMetric`,
`RuntimeDiagnostic`, `CorrelationId`, `TraceId` and `SpanId` as Runtime
contracts.

Rationale: these types provide a stable model without forcing existing
resolution, planning or scheduling code to depend on a third-party telemetry
schema.

Alternative considered: emit backend-specific telemetry directly. Rejected
because it would make the Runtime contract depend on external schemas and would
make Provider privacy harder to preserve.

### Derive events from existing Runtime decisions

Execution plans expose `observations()` derived from selected Providers,
rejected candidates, resource affinity constraints, memory diagnostics and plan
creation. The Scheduler records its own observations for queueing, Provider
submission, execution start, completion, cancellation, failure, interruption and
backpressure.

Rationale: the Runtime already has stable decision records. Deriving events from
those records avoids parallel state and keeps observability aligned with actual
Runtime behavior.

Alternative considered: introduce a global mutable event bus. Rejected for now
because the current Runtime is library-local and deterministic tests benefit
from explicit observation accessors.

### Use one trace for one planned execution

Each `ComputeExecutionPlan` owns a `TraceId`. Scheduler and Provider execution
events reuse that trace when the plan is scheduled and executed.

Rationale: observers can reconstruct one execution across planning and runtime
phases without needing Provider-native identifiers.

Alternative considered: create independent traces per phase. Rejected because
it would make cross-phase reconstruction harder and would require additional
join metadata.

### Keep metrics structured and unit-bearing

`RuntimeMetric` uses stable metric kinds and explicit units. The initial derived
metrics cover memory estimates, transfer volume and materialization count, while
the model also reserves queue, planning and execution latency and Provider
utilization kinds.

Rationale: metrics need stable meaning and units before they can be exported to
multiple sinks.

Alternative considered: expose arbitrary name/value pairs only. Rejected
because it would weaken compatibility for future tooling.

### Model exporters as Components

`ObservabilityExporterDescriptor` describes a WASM Component that consumes the
stable `magnetar:runtime/observability@1.0.0` input contract and writes to an
OpenTelemetry, Prometheus, Jaeger or custom sink.

Rationale: exporters can be added without changing Provider execution or the
Runtime observation contract.

Alternative considered: implement transport-specific exporters in the Runtime.
Rejected because this proposal defines the Runtime contract, not telemetry
transport.

## Risks / Trade-offs

`RuntimeEvent` currently stores messages as redacted strings, so overly broad
redaction can remove useful backend context. Mitigation: stable diagnostic codes
and Runtime identifiers remain available even when message text is redacted.

Observation collection currently lives on explicit plan and scheduler accessors
instead of a process-wide subscriber. Mitigation: this keeps the first contract
deterministic; a future transport change can add dispatch without changing the
observation types.

Latency metric kinds are defined before full timing capture is implemented.
Mitigation: metric kinds are stable, and producers can emit them when timing
sources are added.

## Migration Plan

1. Add Runtime observation types and accessors.
2. Derive observations from execution plans.
3. Record scheduler lifecycle observations.
4. Add architecture documentation and tests for trace correlation, metrics,
   redaction and exporter descriptors.
5. Future exporter implementations can consume the same stable observation
   types without Provider API changes.

Rollback is straightforward because the model is additive. Removing the
observation accessors and event fields does not change Provider execution
semantics.

## Open Questions

- Which timing source should populate queue, planning and execution latency in
  production builds?
- Should a future Runtime instance own a subscriber registry or keep
  observation collection pull-based?
