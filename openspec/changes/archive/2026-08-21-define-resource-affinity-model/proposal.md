# Define Resource Affinity Model

## Why

Magnetar can resolve a compatible Provider before work begins, but it cannot
currently keep later calls on the Provider, Device, capability implementation,
and artifact set that own an opaque resource. Stateful Compute resources make
that gap immediate: independently resolving their dependencies can produce an
execution chain that cannot be consumed safely.

## What Changes

- Introduce host-side Resource Affinity identifiers and bindings for Providers,
  Devices, capability versions, artifacts, execution contexts, and affinity
  groups.
- Add an immutable affinity descriptor, a separate constraint aggregator, and
  a reusable envelope for Provider-owned opaque resources.
- Add structured compatibility errors and affinity-aware Provider resolution
  that resolves all dependencies as one constrained set.
- Classify recovery as transparent, restartable, or Provider-pinned while
  keeping every live resource binding authoritative until explicit recreation
  or transfer.
- Restrict transparent Provider fallback to resolution before incompatible
  Provider-owned state exists.
- Document how the foundation applies to Compute tensor, graph, and operation
  resources and to future model, tokenizer, template, and generation contracts.
- Keep `magnetar:compute@1.1.0` unchanged; concrete WIT resource attachment and
  future model-level WIT contracts require separate changes and host adapters.

## Capabilities

### New Capabilities

- `resource-affinity`: Defines host-side ownership facts, compatibility rules,
  constraint aggregation, structured failures, and constrained resolution for
  opaque runtime resources.

### Modified Capabilities

- `provider`: Limits fallback when existing Resource Affinity pins live state
  to a Provider or Device.

## Impact

- Adds public, additive APIs to `magnetar-runtime` without changing the
  `Provider` trait or `PROVIDER_API_VERSION`.
- Adds runtime-local execution-context and affinity-group identity.
- Adds affinity-aware resolution alongside the existing stateless resolver.
- Adds architecture documentation and tests for Compute and future resource
  chains.
- Adds no dependency and performs no migration of live resources.
