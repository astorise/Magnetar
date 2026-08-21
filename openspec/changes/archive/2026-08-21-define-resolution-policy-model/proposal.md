# Define Resolution Policy Model

## Why

Magnetar resolves Component imports through Capabilities implemented by
Providers.

The Runtime must not simply select the first compatible Provider.

Provider selection depends on:

- capability compatibility
- resource affinity
- device availability
- Provider health
- execution phase
- fallback classification
- user or runtime policy
- cost, latency, memory or energy preferences

The Resource Affinity Model ensures that resources remain bound to coherent
Provider, Device, artifact and Capability chains.

The Resolution Policy Model defines how the Runtime chooses among compatible
implementations before execution begins.

It also defines what the Runtime may and may not do after stateful resources
already exist.

This change does not introduce live state migration.

This change does not introduce automatic execution failover.

It defines the decision model required before Scheduler and execution-health
features can be added safely.

## What Changes

This proposal introduces Resolution Policies as first-class runtime concepts.

A Resolution Policy describes how the Runtime ranks compatible Provider
implementations for a requested Capability.

The policy may consider:

- compatibility
- priority
- Provider health
- Device metadata
- resource affinity
- fallback class
- execution phase
- memory constraints
- latency preference
- throughput preference
- energy preference

The Runtime uses the selected policy during capability resolution.

Resolution Policies are applied before Provider execution begins.

For Provider-pinned resources, the Runtime must preserve the original resource
affinity and must not silently re-resolve the call to another Provider.

## Impact

Provider selection becomes explicit, inspectable and testable.

Fallback behavior becomes phase-aware instead of implicit.

Future Scheduler work can reuse the same policy model for placement, retry,
cost optimization and heterogeneous execution.