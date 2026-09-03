## Context

`transactional-weight-materialization` (archived `2026-09-03`) fixed
weight-materialization ordering, error propagation, and rollback, plus
Provider-owned storage cleanup on unload. A follow-up revalidation audit of
that fix (commit `633e942`) found one remaining gap in the same lifecycle:
`MemoryManager::release(allocation)` only mutates the `MemoryAllocation`'s
own `state` field (`Active` -> `Reusable`/`Released`); it never touches the
separate `tensor_residency: BTreeMap<TensorResourceId, TensorResidency>` map.
Both `WeightMaterializationTransaction::abort` and
`Runtime::unload_model_instance` already release a weight's Provider tensor
and Memory Manager allocation, but neither removes the `TensorResidency`
entry `record_tensor_residency` inserted when the weight was staged. The
entry survives indefinitely, so `MemoryManager::tensor_residency(id)` keeps
returning `Some` for a resource whose Provider storage and allocation are
both already gone.

This is a metadata leak, not a resource leak (Provider storage and Memory
Manager accounting are both already correct after `transactional-weight-
materialization`) -- but it grows once per failed materialization attempt
and once per load/unload cycle, and it is a correctness hazard for any
future caller that treats `tensor_residency()` as evidence a resource
currently exists.

## Goals / Non-Goals

**Goals:**
- After a weight's Provider tensor and Memory Manager allocation are both
  released (rollback or unload), `tensor_residency()` for that resource
  returns `None`.
- The removal happens in the correct order relative to resolving the
  resource's owning Provider: `Runtime::unload_model_instance` reads
  `tensor_residency()` to find the Provider before releasing anything, so
  residency removal must come after that read, not before it.

**Non-Goals:**
- Introducing a richer residency state machine (`Resident` / `Released` /
  `Invalid` / `Evicted`, the audit's "Option B"). The audit itself
  recommends outright removal ("Option A") as "le meilleur choix pour le
  baseline v0.1"; Magnetar does not currently need historical residency
  states, and adding one here would be scope creep beyond the actual gap.
- Touching the separate `resource_residency: BTreeMap<TensorResourceId,
  ResidencySet>` map (a distinct, more general residency mechanism used
  elsewhere in `memory.rs`, e.g. `map_resource`/`readable_residency`) -- the
  audit's finding and this fix are scoped to `tensor_residency`, the map
  weight materialization actually uses.

## Decisions

**Add `MemoryManager::remove_tensor_residency(&TensorResourceId) ->
Option<TensorResidency>`, a plain `BTreeMap::remove`, mirroring
`record_tensor_residency`'s own minimal style (no observation event --
`record_tensor_residency` does not emit one either).** No richer return
type or error case: removing a residency record that does not exist is not
an error (idempotent by construction, matching the audit's "la libération
doit être idempotente et complète" for the sibling Provider/allocation
release calls).

**Rollback order (`WeightMaterializationTransaction::abort`): release
Provider tensor, then remove residency, then release the Memory Manager
allocation.** Matches the audit's own recommended order (section 12). The
Provider release and residency removal are independent of each other here
(both keyed by the same `TensorResourceId`, neither reads the other), so
their relative order does not matter for correctness, but is kept in the
audit's stated order for consistency with the unload path below, where
order *does* matter.

**Unload order (`Runtime::unload_model_instance`): resolve the owning
Provider via `tensor_residency()`, release the Provider tensor, remove the
residency record, then release the Memory Manager allocation.** Order
matters here: the existing code already reads `tensor_residency(resource_id)`
to find which Provider owns the tensor before calling `release_tensor`;
removing the residency record has to happen after that read, not before,
or the Provider resolution would find nothing and silently skip the
`release_tensor` call it already correctly makes today.

## Risks / Trade-offs

- [Risk] A future call site that stages a `TensorResidency` for a resource
  outside `WeightMaterializationTransaction` (e.g. a hypothetical KV cache
  or non-weight tensor path added later) forgets the matching removal on
  its own release path, reintroducing the same class of leak elsewhere. →
  Not mitigated structurally by this fix (no shared "release" helper exists
  that bundles Provider release + residency removal + allocation release
  into one call). Worth revisiting if a third caller of
  `record_tensor_residency` appears; not done here to keep this fix scoped
  to the audit's actual finding.
