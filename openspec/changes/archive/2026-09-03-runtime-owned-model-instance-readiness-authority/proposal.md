## Why

A full-scope audit of PR #36 (commit `a4d411b`) found that `ModelInstance`
readiness, though no longer forced `Ready` on creation, was still partly
caller-declared rather than Runtime-derived: `warm_model_instance`'s public
`ModelInstanceReadinessChecks` parameter defaults `weights_materialized:
true`, and the Runtime trusted it outright, so a caller using default
checks could warm an instance whose weights were never materialized
straight to `Ready`. `WarmupPolicy::Disabled` compounded this: it called
`validate_readiness` without the lifecycle transition warmup's other
policies perform, so it could publish `readiness: Ready` while `lifecycle`
stayed `Loading` -- an internally inconsistent state -- and
`acquire_usage`/`generation_reference` only checked `readiness`, never
`lifecycle`, so that inconsistency was itself usable. Following
clarification with the auditor (the fix must close real forgery paths for
this PR without a full warmup-API redesign), this closes the behavioral
gap without changing the public signature of `warm_model_instance` or
`ModelInstanceReadinessChecks`.

## What Changes

- `warm_model_instance` (`inference_api.rs`) now derives
  `weights_materialized`, `provider_ready`, and `device_ready` from actual
  Runtime state (resource bindings, the Provider registry, the Device
  list) and ANDs them with the caller's claim, instead of trusting the
  caller's `true` outright. A caller MAY still force a stricter `false`;
  MAY NOT force a Runtime-internal fact `true` the Runtime does not itself
  observe.
- `ModelInstance::validate_readiness` no longer allows a computed `Ready`
  readiness to be published while the lifecycle has not actually reached
  a state that supports it (fixes the `WarmupPolicy::Disabled` gap).
- `ModelInstance::acquire_usage` and `ModelInstanceManager::generation_reference`
  now require both `lifecycle.supports_inference_use()` and
  `readiness.accepts_generation()` -- a structural safety net that holds
  regardless of which path might produce an inconsistent state.
- `kernel_preparation_ready`, `autotuning_ready`, `adapter_ready`,
  `memory_pressure`, `runtime_policy_allows`, `residency_available`, and
  `browser_supported` remain caller-supplied: no generic Runtime-side
  derivation for these exists in the current baseline, and inventing one
  now would be scope creep beyond this audit's confirmed findings (see
  design.md).
- Direct access to `ModelInstanceManager::mark_ready`/`ModelInstance::mark_ready`
  remains unchanged (not sealed) -- a deliberate, documented scope
  decision, not an oversight (see design.md's Non-Goals).

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `model-instance`: "Model Instance Readiness", "Model Instance Warmup",
  and "Generation Requires Ready Model Instance" gain explicit
  requirements that readiness-relevant Runtime facts are Runtime-derived,
  not caller-declared, and that usage requires both lifecycle and
  readiness to agree.

## Impact

- `magnetar-runtime/src/inference_api.rs`: `warm_model_instance`,
  new `derive_effective_readiness_checks`.
- `magnetar-runtime/src/model_instance.rs`:
  `ModelInstanceLifecycleState::supports_inference_use`,
  `ModelInstance::validate_readiness`, `ModelInstance::acquire_usage`,
  `ModelInstanceManager::generation_reference`.
- Test coverage: 4 new tests in `magnetar-runtime/src/tests.rs` covering
  the audit's own acceptance scenarios (forged `weights_materialized`,
  the `Disabled`-policy inconsistency, the lifecycle/readiness safety
  net, and the unaffected happy path).
