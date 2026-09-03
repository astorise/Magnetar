## Why

A full-scope revalidation audit of PR #36 (commit `9ec9cd3`) found that
`runtime-owned-model-instance-readiness-authority` (round 1) closed the
specific forgery scenario it demonstrated (default
`ModelInstanceReadinessChecks` against empty `resource_bindings.weights`)
but not the general property it claimed to establish. Three concrete gaps
remained, all with the same root cause -- readiness was still partly
caller-forgeable:

1. `ModelInstanceManager::mark_ready`/`ModelInstance::mark_ready`/
   `transition_to`/`warmup` were still `pub`, directly callable via the
   already-public `Runtime::model_instances_mut()`, completely bypassing
   `warm_model_instance`'s Runtime-derived checks.
2. `ModelInstance.lifecycle`/`.readiness` were still fully public mutable
   fields -- even with (1) closed, `instance.lifecycle = Ready` would have
   remained a working bypass.
3. `weights_materialized` derivation only checked `resource_bindings.
   weights` was non-empty, not that its entries were backed by a real
   `TensorResidency` record -- a caller could insert an arbitrary
   `TensorResourceId` directly and pass the check. `provider_ready`
   similarly only checked a Provider was registered and exposed an
   `execution_api()`, not that its own status model (`status_snapshot()`)
   actually reported it accepting new work.

## What Changes

- `ModelInstance::mark_ready`/`transition_to`/`warmup` and the matching
  `ModelInstanceManager` wrappers become `pub(crate)`. The only public path
  to `Ready` is `warm_model_instance` (`inference_api.rs`).
- `ModelInstance.lifecycle`/`.readiness` become `pub(crate)` fields (not
  fully private: this crate's own test suite legitimately needs to
  construct otherwise-unreachable states to prove defense-in-depth checks
  hold), with public read-only accessor methods `lifecycle()`/
  `readiness()` for external callers.
- `derive_effective_readiness_checks` (`inference_api.rs`) now requires
  every bound weight to have a matching `TensorResidency` record, not just
  a non-empty map, and derives `provider_ready` from
  `Provider::status_snapshot().accepts_new_work_by_default()` in addition
  to `execution_api().is_some()`.
- `magnetar-runtime/tests/contract_tests/model_instance.rs` (an external
  consumer of the crate's public API, same as any embedder) is migrated
  off the now-crate-internal primitives onto `warm_model_instance` with
  real evidence -- this is not a workaround, it is the same contract a
  real embedder must now follow.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `model-instance`: "Model Instance Readiness" and "Model Instance Warmup"
  gain explicit requirements that the public surface cannot produce
  `Ready` without Runtime-verified evidence, and that `weights_materialized`/
  `provider_ready` derivation consults real backing state (residency,
  Provider status), not just presence/existence checks.

## Impact

- `magnetar-runtime/src/model_instance.rs`: field/method visibility.
- `magnetar-runtime/src/inference_api.rs`: strengthened derivation.
- `magnetar-runtime/tests/contract_tests/model_instance.rs`: migrated to
  the public `warm_model_instance` contract.
- `magnetar-runtime/src/tests.rs`: 2 new dedicated tests for the
  strengthened derivations; 1 existing test fixed (it had the exact
  fake-weight-no-residency gap this Change closes).
