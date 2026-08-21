# Resolution Policy

Resolution policy is the runtime decision model for choosing one Provider and
one exact Capability implementation before execution begins.

It does not migrate live state. It does not add automatic execution failover.
Resources that already carry Provider affinity remain bound to that Provider
until the caller explicitly releases, cancels, interrupts, fails, or recreates
them.

## Lifecycle

1. Convert a Component import or host request into a `CapabilityId` and required
   `CapabilityVersion`.
2. Aggregate any dependent `ResourceAffinity` values and reject conflicts before
   Provider invocation.
3. Build `ResolutionCandidate` values from compatible Provider, Capability and
   Device metadata.
4. Apply the active `ResolutionPolicy` from `RuntimeConfig`.
5. Return a `ResolutionDecision` with stable diagnostics: selected Provider,
   selected Device when known, selected Capability version, policy id, reason,
   and candidate rejection categories.
6. Attach the selected Provider, Capability version and Runtime execution
   context to new resource affinity.

For Provider-pinned dependent calls, the candidate set is intentionally limited
to the Provider already recorded in affinity. The Runtime validates that the
Provider still exists and still implements the needed Capability; it does not
silently choose another Provider.

## Built-in Policies

- `Deterministic`: selects by stable Provider, Capability and Device identity.
- `Priority`: reserved for explicit priority metadata; currently falls back to
  deterministic ordering when priorities are equal.
- `Availability`: prefers healthier Providers and available Devices, then uses
  deterministic ordering.
- `PerformancePreferred`: placeholder for future latency or throughput
  scoring, with deterministic fallback today.
- `EnergyPreferred`: placeholder for future energy scoring, with deterministic
  fallback today.
- `MemoryConstrained`: placeholder for future memory-fit scoring, with
  deterministic fallback today.

All built-in policies reject unavailable Providers, unavailable Devices,
affinity-incompatible candidates, and fallback attempts that are not safe for
the current execution phase.

## CPU and GPU Selection

A Provider may expose CPU and GPU Devices with execution-capability metadata.
When no live affinity is present, the Runtime builds candidates from compatible
Providers and available Devices. `Availability` can prefer a healthy GPU-backed
candidate over a degraded CPU-backed Provider, but the final decision still
records stable identifiers rather than native handles.

Once a tensor, graph or operation is created, its `ResourceAffinity` records the
selected Provider and exact `magnetar:compute/run` version. Dependent Compute
calls aggregate those affinities and preserve the existing Provider chain.

## Generation Session Pinning

A generation session owns Provider state such as loaded model handles, KV cache,
RNG state and streaming sequence. Its affinity should be classified as
Provider-pinned after session creation.

If the owning Provider becomes unavailable after output has started, the Runtime
reports interruption or failure. It does not continue the same session on a
different Provider. A caller may create an explicit new session only when its
application-level replay policy can tolerate different model execution.

## Transparent Versus Restartable

Transparent selection is only for work before observable state exists. For
example, if a preferred Provider is unavailable before a tensor is allocated or
before a generation session is created, policy may choose another compatible
Provider.

Restartable selection requires replayable input and must not duplicate
observable output. It may apply to a request that failed before returning any
output, but it is rejected after observable output has been emitted.

Provider-pinned live state is stricter than both. It requires explicit teardown
or failure handling rather than implicit re-resolution.
