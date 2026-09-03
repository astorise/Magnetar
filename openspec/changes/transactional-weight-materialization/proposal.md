## Why

An external audit of PR #36 (commit `0197be1`, the commit that declared Architecture Freeze #1 accepted) correctly identified two real, verified P0 defects in the weight-materialization lifecycle: a Model Instance is marked `Ready` immediately on creation, before mandatory weights are ever bound, and `acquire_usage`'s own readiness check does not inspect weight bindings at all; and weight materialization is not transactional -- `Provider.write_tensor` is called before `MemoryManager.allocate` (inverting the intended admission-before-materialization order), a residency-registration error is silently discarded, a mid-loop failure leaves already-written weights un-rolled-back, and `unload_model_instance` releases `MemoryAllocationId`s but never releases the Provider-owned weight `TensorResourceId`s themselves, leaking Provider storage across every load/unload cycle. Both were verified directly against the current code, not assumed from the audit's description. Architecture Freeze #1 has been reverted to `candidate` pending this fix.

## What Changes

- `ModelInstances::create()` no longer calls `mark_ready()` unconditionally; a freshly created instance stays in `Loading` (non-Ready) state. A new `ModelInstances::mark_ready(id)` method (mirroring the existing `warmup()` method's pattern) performs the real `Loading -> Ready` transition and its observation, called explicitly once weight materialization succeeds.
- `materialize_model_instance_weights_inner` is rewritten as a `WeightMaterializationTransaction`, mirroring the existing, already-correct `KvUpdateTransaction` pattern (task 9's Correctif 11 fix): each weight is staged via Memory Manager admission *then* Provider write *then* residency registration (with the residency error now propagated, not discarded via `let _ =`); on any failure, every weight staged so far in the same attempt is rolled back (Provider tensor released, Memory Manager allocation released); only after every weight stages successfully does the transaction commit, publish bindings, and mark the instance Ready.
- `Runtime::unload_model_instance` releases the Provider-owned weight Tensor Resources bound in `resource_bindings.weights`, not only the Memory Manager allocations, resolving the Provider that owns them the same way materialization does.
- Corrects `materialize-weights-from-real-model-artifact`'s "Weight Resource Completeness Gates Generation" requirement in place: its original wording explicitly permitted a Model Instance to report Ready before materialization, relying on a deeper dispatch-time check alone -- the property this audit shows is not a sufficient guarantee.
- Updates and adds tests per the audit's own specified list: instance not-Ready immediately after creation; generation-before-materialization rejected; full materialization reaches Ready; a failure at any point in the weight loop never produces a Ready instance and rolls back every prior weight in that attempt; residency-registration failure is propagated and rolls back; load-then-unload returns Provider storage to baseline; repeated load/unload cycles do not accumulate Provider storage.
- Files P1 GitHub issues (not fixed in this Change) for the audit's three P1 findings: the generic graph executor only propagating a multi-output node's first output past `ReferenceCpuExecutor` itself, `model-family-isolation`'s CI guard allowlist not yet covering every generic Core file, and embedder-pushed Component artifacts being classified `DevelopmentFixture` instead of `ClientProvided`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-instance`: "Model Instance Creation" gains an explicit normative statement that creation alone SHALL NOT produce a Ready instance -- closing the ambiguity that let the current bug pass unnoticed (the requirement previously said nothing either way).

The other requirement this fix corrects -- "Weight Resource Completeness Gates Generation, Not Merely Instance Lifecycle" (`model-loading`) -- is edited in place on `materialize-weights-from-real-model-artifact`, the change that first introduced it, per this repository's own convention of not leaving a known-wrong requirement live under a different change's name; see that change's updated spec delta and design.md correction note.

## Impact

- `magnetar-runtime/src/model_instance.rs`: `ModelInstances::create()` no longer auto-readies; new `ModelInstances::mark_ready()`.
- `magnetar-runtime/src/first_native_runtime.rs`: `materialize_model_instance_weights`/`materialize_model_instance_weights_inner` rewritten around a new `WeightMaterializationTransaction`; `check_weight_materialization_failure_demotes_instance_from_ready` rewritten to test "never becomes Ready" instead of "becomes Ready then is demoted".
- `magnetar-runtime/src/runtime.rs`: `unload_model_instance` releases Provider-owned weight tensors.
- `magnetar-runtime/src/tests.rs`: two existing tests (`inference_api_model_instance_warmup_reports_lifecycle_conflict_when_already_ready`, `inference_api_model_instance_suspend_resume_drain_through_api_boundary`) updated to call the new explicit `mark_ready()` step instead of relying on `create()`'s old auto-ready behavior.
- `CHANGELOG.md`: Architecture Freeze #1 stays `candidate` until this Change lands, is verified, and a fresh CI run plus live `magnetar run qwen-test "Hello"` confirm it -- not re-declared as part of this Change itself.
- No new external dependencies; no WIT contract changes; no Component-facing API changes.
