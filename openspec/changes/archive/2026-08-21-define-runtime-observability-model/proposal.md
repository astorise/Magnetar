# Define Runtime Observability Model

## Why

Magnetar now owns every major execution decision:

- Capability Resolution
- Provider Selection
- Resource Affinity
- Memory Planning
- Execution Planning
- Scheduling
- Provider Execution
- Provider Health

The Runtime must expose those decisions through a stable observability model.

Observability is not debugging.

Observability provides portable insight into Runtime behavior while preserving
Provider implementation privacy.

Components, tools and administrators need stable metadata describing:

- what happened
- when it happened
- why it happened
- which Runtime object was involved

without exposing backend implementation details.

## What Changes

This proposal introduces the Runtime Observability Model.

The Runtime produces structured observations for:

- Capability Resolution
- Resource Affinity
- Execution Planning
- Scheduling
- Provider Execution
- Data Movement
- Memory Planning
- Provider Health
- Runtime Errors

Observations contain stable Runtime identifiers.

Observations never expose:

- raw pointers
- queues
- streams
- native handles
- backend storage
- Provider-private objects

The model supports:

- tracing
- metrics
- events
- structured diagnostics
- correlation identifiers

This proposal does not define any telemetry transport.

This proposal does not mandate OpenTelemetry.

This proposal defines only the Runtime contract.

## Impact

Every Runtime decision becomes observable.

External tooling can reconstruct execution without depending on Provider
internals.

Future OpenTelemetry, Prometheus, Jaeger or custom exporters can be implemented
without modifying Runtime contracts.

## Exporter Components

Runtime observations MAY be consumed by WASM Components.

Exporter Components MAY transform Runtime observations into external telemetry
formats such as OpenTelemetry, Prometheus, Jaeger or custom formats.

Exporter Components SHALL consume stable observability WIT contracts only.

Exporter Components SHALL NOT access Provider-native handles, backend storage,
queues, streams, raw pointers, credentials or ambient filesystem paths.