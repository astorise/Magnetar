# Prepared Execution Plan

Prepared Execution Plans are Runtime-owned execution recipes derived from a
portable `ExecutionGraph`. The graph remains the semantic source of truth; a
plan records selected execution decisions for one compatible workload context.

## Execution Graph Versus Prepared Plan

`ExecutionGraph` describes portable Operator semantics, topology, attributes,
and logical tensor descriptors. `PreparedExecutionPlan` references the graph
semantic fingerprint and binds validated graph nodes to exact Kernels,
specializations, Provider/Device choices, resource slots, guards, and optional
Provider-prepared segments.

The plan must not add, remove, reorder, fuse, or reinterpret operations unless
the graph and Operator contracts already permit the transformation.

## Lifecycle

Plan generations move through:

```text
building -> validating -> preparing -> ready
ready -> stale -> invalidated
ready/stale/invalidated -> retiring -> retired
```

`failed` terminates unsuccessful preparation. `ready` and policy-allowed
`stale` plans may accept new work. `invalidated`, `retiring`, `retired`, and
`failed` plans must not accept new work.

## Guards

Guards are cheap Runtime checks evaluated before dispatch. They cover shape
envelopes, dtype, layout, execution phase, batch and sequence bounds,
continuous-batch constraints, adapter revision, KV layout, affinity, Provider
readiness, Device readiness, and memory feasibility.

Guard evaluation must not perform Registry discovery, qualification,
benchmarking, compilation, or full memory replanning.

## Resource Slots

Plans describe resources by logical slots:

```text
input:hidden-state
output:logits
model:weights
workspace:attention
session:kv-key
session:kv-value
```

Stable resources may refer to model weights, immutable adapter weights, or
Provider-prepared constants. Dynamic resources are bound per invocation,
session, or batch quantum. Plans do not own Runtime tensor memory; the Memory
Manager remains authoritative for allocation, residency, eviction, movement,
and lifetime.

## Stale Versus Invalidated

`stale` means the plan remains safe and eligible but may no longer be optimal.
Typical causes are Kernel promotion, preference-only policy updates, stale
autotuning evidence, performance regression, and workload drift.

`invalidated` means the plan is no longer safe or policy-eligible for new work.
Typical causes are Kernel revocation, qualification revocation, trust denial,
Provider or Device unavailability, affinity incompatibility, incompatible
Model Instance revision, missing Prepared Kernel state, memory infeasibility,
or hard policy change.

## Plan Families

A Model Instance may hold plan families for one graph fingerprint. Families are
keyed by graph fingerprint, Model Instance revision, phase, and workload
bucket. This supports distinct prefill/decode plans, multiple shape envelopes,
and cheap bounded lookup for compatible work.

## Atomic Replacement

A replacement generation is fully prepared and marked ready before publication.
Publishing swaps the active generation atomically for new work, marks the old
generation retiring, and keeps old Provider-prepared state alive while leases
exist for in-flight invocations.

## Provider Prepared Segments

Providers may advertise optional prepared-segment support such as native graph
capture or command sequence preparation. Runtime owns the logical segment; the
Provider owns native prepared state and returns only an opaque
`ProviderPreparedSegmentId`. Providers without segment capture remain valid and
dispatch individual prepared Kernels.

## Cache And Restart

Plan cache entries are distinct from Kernel Artifact, Autotuning, Model
Artifact, Prefix, and KV caches. Cache keys include graph fingerprint, Model
Instance revision, workload scope, Kernel artifact digests, specialization IDs,
Provider/Device compatibility, policy versions, memory-plan version, adapter
revision, and KV layout.

Persisted plan data is a logical recipe. Runtime restart must recheck
revocation, trust, qualification, Provider readiness, Device readiness,
Prepared Kernel reconstruction, and memory feasibility before a cached plan can
be ready.

## Hot Path Objective

The objective is bounded Runtime decision overhead, not zero overhead. Ready
plan execution performs plan lookup, guard validation, resource binding, lease
acquisition, prepared segment or Kernel dispatch, observation, and lease
release without rebuilding the full Kernel resolution pipeline.
