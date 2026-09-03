## Context

`ModelInstances::create()` (`magnetar-runtime/src/model_instance.rs`) currently does:

```rust
let mut instance = ModelInstance::new(id.clone(), definition);
instance.transition_to(ModelInstanceLifecycleState::Loading)?;
instance.mark_ready()?;
self.instances.insert(id.clone(), instance);
self.observe(ModelInstanceObservationKind::Created, ...);
self.observe(ModelInstanceObservationKind::Ready, ...);
Ok(id)
```

Weight materialization (`materialize_model_instance_weights_inner`, `magnetar-runtime/src/first_native_runtime.rs`) runs entirely *after* this, as a separate call. Its current body, per weight:

```rust
executor.write_tensor(resource_id.clone(), tensor.clone());          // Provider write, unconditional
let allocation = runtime.memory_mut().allocate(...)?;                // Memory admission, AFTER the write
let _ = runtime.memory_mut().record_tensor_residency(...);           // error discarded
// ... resource_bindings.weights.insert(...) unconditionally follows
```

`Runtime::unload_model_instance` releases `report.released_memory_allocations` (Memory Manager bookkeeping) but never calls `Provider.release_tensor` for the weight `TensorResourceId`s recorded in `resource_bindings.weights`.

An external audit of PR #36 (commit `0197be1`) found all of this by reading the real code, not by inference from behavior, and it is confirmed accurate. A near-identical problem was already solved correctly once in this codebase, for a different resource class: `KvUpdateTransaction` (task group 9, Correctif 11) admits-then-writes each KV layer, tracks what has been staged, and provides `abort`/`commit` that release or publish everything staged in one atomic unit. This design is "do for weights what task 9 already did for KV," not a new pattern.

## Goals / Non-Goals

**Goals:**
- A Model Instance's own reported lifecycle/readiness is trustworthy on its own -- `acquire_usage`-style checks that only look at the coarse flag are safe, not merely "safe because something deeper also happens to check."
- Weight materialization admits through the Memory Manager before writing to Provider-owned storage, for every weight.
- A residency-registration failure is a real, propagated error.
- A failure at any point in materializing a multi-weight instance rolls back every weight already staged in that same attempt -- no partial state survives.
- Unloading a Model Instance returns Provider-owned storage to baseline, not just Memory Manager accounting.

**Non-Goals:**
- Changing `ModelLoadingCoordinator::load()`'s signature or the Lazy Loading Policy. That question was already explicitly decided (twice, across two prior Changes) to keep materialization a distinct step after `load()`, and the audit's own recommendation (section 9) explicitly agrees this separation should be kept: `load() != materialize()` remains correct. Only *when the instance reports Ready* changes, not *when materialization runs relative to `load()`*.
- Making `ProviderExecutionApi::write_tensor` itself fallible. A real Provider's write could conceivably fail (out of device memory, transfer error), and today's trait method has no error channel for that at all -- a genuine, separate gap. `write_tensor_admitted`-style admission-first sequencing (used here, matching `KvUpdateTransaction`'s own pattern of `memory.allocate()` then `executor.write_tensor()`) sequences correctly around this without requiring a breaking trait signature change. Making `write_tensor` fallible for every Provider implementation (including the external `providers/cpu`/`providers/cuda` submodules) is real, separate, larger work, filed as a P1 follow-up issue rather than done here.
- The audit's three P1 findings (multi-output graph executor propagation, `model-family-isolation` allowlist completeness, `DevelopmentFixture` vs `ClientProvided` classification for embedder-pushed artifacts). Filed as GitHub issues per this repository's own governance rule (spec-correct-but-code-nonconformant issues get a GitHub issue, not a new Change) -- none of the three block re-accepting Architecture Freeze #1, per the audit's own verdict (section 2: only the two P0s block acceptance).

## Decisions

**`ModelInstances::create()` stops calling `mark_ready()`; a new `ModelInstances::mark_ready(id)` performs the real transition, mirroring the existing `warmup()` method exactly.** `warmup()` already establishes the pattern this fix reuses: call into the individual `ModelInstance`'s own state-machine method, then emit the matching observation based on the real outcome.

```rust
// model_instance.rs, ModelInstances
pub fn mark_ready(&mut self, id: &ModelInstanceId) -> Result<(), ModelInstanceError> {
    let result = self.instance_mut(id)?.mark_ready();
    self.observe(
        if result.is_ok() { ModelInstanceObservationKind::Ready } else { ModelInstanceObservationKind::Failed },
        Some(id.clone()),
        if result.is_ok() { "model instance ready" } else { "model instance mark-ready failed" },
        None,
    );
    result
}
```

`ModelInstance::mark_ready()` itself (the individual-instance method) is unchanged -- it already correctly handles the `Creating|Loading|Warming -> Ready` transition via `transition_to`. The only change is *who calls it and when*: `create()` no longer calls it inline; the weight-materialization success path does, explicitly, after real success.

*Alternative considered:* route readiness through the existing `validate_readiness(checks: &ModelInstanceReadinessChecks)` mechanism instead, setting `weights_materialized` on the checks struct (that field already exists and is already correctly wired into `readiness()`/`validate()` -- it was simply never invoked from this call site). Rejected for this specific transition: `validate_readiness` sets the `readiness` field but does not itself drive the `lifecycle` state machine (`Loading -> Ready`), and mixing the two per-call would leave `lifecycle` and `readiness` divergent unless every caller remembers to also call `transition_to`. `mark_ready()` already does both correctly in one call. `ModelInstanceReadinessChecks.weights_materialized` remains available and correct for its existing use (`warmup()`'s own richer, multi-condition readiness evaluation); this fix does not touch or need it.

**`materialize_model_instance_weights_inner` becomes `WeightMaterializationTransaction`, structured identically to `KvUpdateTransaction`.**

```rust
struct WeightMaterializationTransaction {
    provider_binding: ProviderBinding,
    executor: Arc<dyn ProviderExecutionApi>,
    staged: Vec<StagedWeight>,
}

struct StagedWeight {
    name: String,
    resource_id: TensorResourceId,
    allocation: MemoryAllocationId,
}

impl WeightMaterializationTransaction {
    fn begin(runtime: &Runtime) -> Result<Self, InferenceApiError> { /* resolve executor, like KvUpdateTransaction::begin */ }

    /// Admits through the Memory Manager, THEN writes to Provider storage,
    /// THEN registers residency -- in that order, each step's error
    /// propagated. Does not touch the Model Instance's resource_bindings;
    /// that is commit's job, once the whole attempt has succeeded.
    fn stage_weight(&mut self, runtime: &mut Runtime, artifact_owner: &str, name: &str, tensor: &HostTensor) -> Result<(), InferenceApiError> { ... }

    /// Releases every resource staged so far this attempt: Provider tensor
    /// then Memory Manager allocation, for each staged weight, in reverse
    /// order (mirroring KvUpdateTransaction::abort).
    fn abort(self, runtime: &mut Runtime) { ... }

    /// Reached only once every weight staged successfully. Publishes every
    /// staged weight's binding onto the Model Instance's resource_bindings,
    /// then marks the instance Ready via ModelInstances::mark_ready.
    fn commit(self, runtime: &mut Runtime, instance: &ModelInstanceId) -> Result<(), InferenceApiError> { ... }
}
```

Caller (`materialize_model_instance_weights`):

```rust
fn materialize_model_instance_weights(runtime: &mut Runtime, instance: &ModelInstanceId, artifact_owner: &str, weights: &BTreeMap<String, HostTensor>) -> Result<(), InferenceApiError> {
    let mut transaction = WeightMaterializationTransaction::begin(runtime)?;
    for (name, tensor) in weights {
        if let Err(error) = transaction.stage_weight(runtime, artifact_owner, name, tensor) {
            transaction.abort(runtime);
            if let Ok(model_instance) = runtime.model_instances_mut().instance_mut(instance) {
                let _ = model_instance.transition_to(ModelInstanceLifecycleState::Failed);
            }
            return Err(error);
        }
    }
    transaction.commit(runtime, instance)
}
```

This is the exact same shape as `promote_pending_kv_resources`/`KvUpdateTransaction` (loop stages, abort-and-return on first error, commit only after the whole loop succeeds) -- no new control-flow pattern, reusing a design this codebase already trusts.

*Alternative considered:* use `ProviderExecutionApi::write_tensor_admitted` (already exists, already does admission-then-write atomically per resource) instead of the manual `allocate()` + `write_tensor()` sequence `KvUpdateTransaction` uses. Rejected for consistency, not correctness -- both orderings are equally correct; `write_tensor_admitted` bundles admission and write into one call with no room for the residency-registration step in between, while `stage_weight` needs that third step. Matching `KvUpdateTransaction`'s existing three-call shape (already proven, already read by whoever maintains this file next to the KV path) was judged more valuable than saving one call via a helper this file does not otherwise use.

**Residency registration's error is propagated, and its failure rolls back the same as any other step's failure -- no special-casing.** `record_tensor_residency` can only fail if the allocation handle it references does not exist in the Memory Manager's own table (`MemoryError::InvalidAllocationHandle`) -- structurally near-impossible immediately after this same transaction just created that exact allocation, but "near-impossible" is not "impossible," and swallowing it via `let _ =` was exactly what the audit correctly flagged (section 5.3). `stage_weight` propagates it with `?` like every other step.

**Unload releases Provider-owned weight storage by resolving each resource's own recorded Provider affinity, not a hardcoded provider name.** `runtime.rs` is generic Core -- it must not import `first_native_runtime.rs`'s `REFERENCE_CPU_PROVIDER_NAME` or its private `resolve_kernel_execution_provider` helper (a different module, and Qwen/first-native-scoped by convention, per task group 12's own CI guard). Every weight resource's `TensorResidency` (recorded by `WeightMaterializationTransaction::stage_weight`'s `record_tensor_residency` call, via `ResourceAffinity::with_provider(...)`) already names its owning Provider generically -- `MemoryManager::tensor_residency(&TensorResourceId) -> Option<&TensorResidency>` and `TensorResidency.affinity.provider() -> Option<&ProviderBinding>` are both already public, existing accessors. Unload uses them:

```rust
// Runtime::unload_model_instance, before releasing memory allocations
for resource_id in &report.released_weight_resources {
    if let Some(provider_binding) = self.memory.tensor_residency(resource_id).and_then(|r| r.affinity.provider())
        && let Some(executor) = self.providers.provider(provider_binding.as_str()).and_then(|p| p.execution_api())
    {
        executor.release_tensor(resource_id);
    }
}
```

This resolves correctly for any Provider a weight happens to be materialized against, not only Reference CPU, and needs no new generic-Core knowledge of any specific Provider name.

`ModelInstanceUnloadReport` needs a new field, `released_weight_resources: BTreeSet<TensorResourceId>`, populated by `ModelInstances::unload()`'s existing `prepare_unload_report` step from the instance's `resource_bindings.weights` values before the instance is removed -- mirroring how `released_memory_allocations` is already populated from `resource_bindings.memory_allocations` in that same step. `release_tensor` is already idempotent-safe (`ReferenceCpuExecutor::release_tensor` returns `bool`, `true`/`false` for present/absent, never panics on a second call), so calling it here even for an instance whose weights were never fully materialized (still empty) is harmless.

**Known, deliberately out-of-scope edge case found while designing this: an instance still in `Loading` (created but never materialized, or abandoned mid-materialization) cannot currently be unloaded at all.** `ModelInstanceLifecycleState::allows_transition_to` has no `Loading -> Draining/Unloading` transition; `ModelInstances::unload()` only proceeds from `Ready | Idle | Suspended | Failed | Invalid`. This is a pre-existing gap, not introduced by this Change -- `Loading` already existed as a state before this fix, just rarely observed in practice because every real caller reached `Ready` near-instantaneously. This fix makes `Loading` a real, potentially longer-lived state, which makes the gap more likely to matter in practice, but closing it (deciding whether `Loading` should be cancelable, and what "unloading a never-materialized instance" should release) is a separate, smaller follow-up, not required to fix either P0 the audit named. Filed as a P2 follow-up issue.

*Alternative considered:* have `WeightMaterializationTransaction::abort`'s per-weight release logic and unload's release logic share one helper. Not pursued as a hard requirement in this Change -- both call `executor.release_tensor(&resource_id)` in a loop already; a shared helper is a small, optional follow-up refactor, not required for correctness, and forcing it now would touch more call sites than this fix needs to.

## Risks / Trade-offs

- [Risk] Two existing tests (`inference_api_model_instance_warmup_reports_lifecycle_conflict_when_already_ready`, `inference_api_model_instance_suspend_resume_drain_through_api_boundary`) assert `create()` produces an immediately-`Ready` instance. → Mitigation: both updated to call the new explicit `mark_ready()` step first; their actual test intent (warmup-on-already-Ready conflict; suspend/resume/drain from Ready) is unaffected, only how "Ready" is reached changes. Confirmed via a full-crate search that no other test or production call site depends on `create()`'s old auto-ready behavior (`create_model_instance`'s only production callers are `first_native_runtime.rs`'s three call sites, all of which already call a materialization function immediately after -- exactly the sequence this fix wires up correctly).
- [Risk] A Provider's real `write_tensor` failing has no error channel today (`ProviderExecutionApi::write_tensor` returns `()`). → Not fixed here (Non-Goal, above); the transaction's ordering still protects the Memory Manager ledger even though a hypothetical Provider write failure can't currently be observed by this specific method. Filed as a P1 follow-up.
- [Trade-off] `WeightMaterializationTransaction` and `KvUpdateTransaction` are structurally near-identical but not merged into one generic abstraction. Accepted: they operate on different resource shapes (N independently-named weights vs. paired K/V per layer) and different publish targets (`ModelInstance.resource_bindings.weights` vs. `KvCache.layer_resources`); forcing a shared generic transaction type now is speculative generalization for two call sites, not something either one currently needs.

## Migration Plan

No data migration -- this changes in-process Runtime state machine behavior and Provider resource lifecycle only, nothing persisted across process restarts. Rollout is: land the code change, run the full verification cycle (fmt/clippy/tests/wasm32/coverage), push, confirm CI green, run `magnetar run qwen-test "Hello"` live, then re-declare Architecture Freeze #1 accepted in `CHANGELOG.md` citing this Change and the fresh CI run -- matching the audit's own acceptance criteria (section 29).
