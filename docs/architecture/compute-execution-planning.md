# Compute Execution Planning

Compute execution planning is the Runtime boundary between graph validation and
Provider submission. It turns a validated `ComputeGraph` into a
`ComputeExecutionPlan` that can be inspected before a Scheduler or Provider
receives work.

An execution plan records the selected Provider, selected Device when one is
available, selected compute Capability, Resolution Policy decision, graph
inputs and outputs, Resource Affinity constraints, required data movement,
required materialization, a `MemoryPlan`, ordered execution steps and stable
diagnostics.

Components do not construct Provider-specific execution plans. Providers do not
select themselves. The Runtime owns planning because it has the portable graph,
resource affinity metadata, Provider advertisements, Resolution Policy and
memory planning state needed to make one coherent decision.

## Lifecycle

1. Validate the `ComputeGraph`, tensor descriptors and operation schemas.
2. Merge input `ResourceAffinity` into planning constraints.
3. Resolve the compute Provider through the active `ResolutionPolicy`.
4. Select a compatible Device from affinity constraints, policy output or
   registered Provider devices.
5. Validate Provider compute advertisements for capability version, operation
   schema, dtype, layout, precision and data movement support.
6. Build or validate the `MemoryPlan`.
7. Record explicit transfer, upload, download, copy and materialization steps
   required by resource placement or view semantics.
8. Validate that all plan dependencies are resolved and that no hidden Provider
   migration or hidden CPU staging has been introduced.
9. Hand the validated plan to the Runtime Scheduler or Provider-submission code.

## Resolution Policy

Execution planning uses the Runtime's active `ResolutionPolicy`. The policy
chooses among compatible Provider candidates after resource affinity, Provider
health, Device availability and capability version gates are applied.

Provider-pinned or Device-bound input resources constrain the candidate set.
When a resource is bound to a Provider, planning preserves that Provider unless
an explicit supported transfer is represented in the plan. Planning does not
silently migrate live Provider-owned state.

## Memory Planning

Every `ComputeExecutionPlan` includes a `MemoryPlan`. Memory planning accounts
for graph inputs, outputs, intermediates, reusable buffers, materialization
buffers, transfer buffers, host staging and Provider or Device memory limits.

Execution planning treats memory failures as structured planning failures before
scheduling. This keeps peak memory, temporary buffers, transfer buffers,
materialization memory and output allocation requirements visible before any
Provider execution begins.

## Scheduler Relationship

The Scheduler consumes validated `ComputeExecutionPlan` values rather than
re-resolving Provider, Device, memory or affinity decisions.

Scheduling may decide when work runs. It must not rewrite the selected Provider,
selected Device, memory requirements or affinity constraints without producing a
new validated execution plan.

## Examples

CPU/GPU selection starts with Resolution Policy. If a graph has no pinned input
resources, the deterministic policy selects the first compatible Provider in
stable order, while availability-oriented policies can prefer healthy
candidates. If the selected Provider has an available registered GPU Device,
the plan records that Device and validates memory against its capacity when the
Device advertises one.

Provider-pinned resources constrain selection. If a tensor input is pinned to
`provider-b`, planning preserves `provider-b` even when `provider-a` also
implements `magnetar:compute/run`. If the graph cannot run on `provider-b` and
no explicit transfer is supported and represented, planning returns a structured
execution-planning error instead of moving the resource implicitly.
