## Why

GitHub issue "A Model Instance stuck in Loading cannot currently be
unloaded or canceled" (filed while designing `transactional-weight-
materialization`: since creation no longer auto-readies, an instance
whose owning caller crashes, is canceled, or never calls
`materialize_model_instance_weights` stays in `Loading` with, the issue
claimed, no explicit way out) asked whether `Loading` should be
cancelable and, if so, for a lifecycle transition and/or unload
entrypoint change plus a canonical spec update recording the decision.

Investigation before writing this proposal found the capability already
exists and is already correct: `ModelInstance::fail`/`invalidate`
unconditionally set lifecycle to `Failed`/`Invalid` regardless of the
instance's current state -- they do not consult
`ModelInstanceLifecycleState::allows_transition_to` at all, unlike every
other lifecycle-mutating method -- and `(Failed, Unloading)` is already
one of `allows_transition_to`'s valid pairs, which
`ModelInstanceManager::unload` already accepts. `runtime.
model_instances_mut().fail_instance(&id, reason)` followed by
`runtime.unload_model_instance(&id, policy)` already takes a `Loading`
instance all the way to `Unloaded`, verified directly by a new test
(`loading_instance_can_be_canceled_via_fail_then_unload`) that creates
an instance, deliberately never materializes it, fails it, unloads it,
and confirms a clean report (no weight resources or memory allocations
released, since none were ever bound) and a final `Unloaded` state.

No code changes were needed. What was actually missing, per the issue's
own Definition of Done ("the decision ... is reflected in `openspec/
specs/model-instance/spec.md`"), is exactly that: this capability was
real but not canonically documented, so this Change adds the one
requirement recording it.

## What Changes

- No production code changes.
- One new regression test proving the existing `fail_instance` +
  `unload_model_instance` path cancels a `Loading` instance cleanly.
- `model-instance` spec gains a new requirement documenting that a
  Model Instance in `Loading` (or any other non-terminal state) can be
  explicitly failed and then unloaded, and that unloading an instance
  that never had any resources bound reports an empty, not erroring,
  release set.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `model-instance`: new requirement, "Loading Model Instance Can Be
  Canceled," alongside the existing "Model Instance Unload" and "Model
  Instance Failure Categories" requirements.

## Impact

- `magnetar-runtime/src/first_native_runtime/tests.rs`: one new test.
- `openspec/specs/model-instance/spec.md`: one new requirement.
