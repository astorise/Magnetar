## Context

This follows a back-and-forth with the external auditor after their audit
of commit `a4d411b` (full re-audit following `transactional-weight-
materialization` and `invalidate-tensor-residency-on-release`) found a new
P0: readiness is still partly caller-declared. The auditor's own scoping
reply (clarifying a section 16 that mixed "necessary for this PR" with
"desirable future architecture") set the exact boundary this Change
implements:

**DO NOW (blocking this PR):**
1. A caller can no longer transition a Model Instance to Ready solely by
   asserting Runtime facts.
2. The real weight/resource state is verified before Ready.
3. Other already-Runtime-observable facts (Provider/Device readiness) are
   not simply trusted because the caller set `true`.
4. `acquire_usage` refuses an incompatible lifecycle, even if readiness is
   accidentally `Ready`.
5. The direct public path to `mark_ready`/`Ready` is sealed *enough* to
   prevent an embedder from bypassing 1-4.
6. Dedicated tests.
7. Green CI.

**NOT required now:** a `WarmupRequest` replacing `ModelInstanceReadinessChecks`,
a full Inference API redesign, removing all mutable access to
`ModelInstance`, deriving future autotuning/adapter signals, or reworking
first-native architecture.

## Goals / Non-Goals

**Goals:** items 1-4 above, with real derivation (not fabricated checks)
for every fact the Runtime can actually observe today.

**Non-Goals:**
- Deriving `kernel_preparation_ready` or `autotuning_ready` from Runtime
  state. Verified directly: no code path today links a `ModelInstanceId`
  to its expected prepared Kernel set or an autotuning signal generically
  (`grep` for `kernel_preparation_ready` outside `model_instance.rs` finds
  nothing). Fabricating a check here would either be a no-op (always true)
  or require new plumbing the auditor explicitly said not to invent now.
  Left caller-supplied, documented in code and here.
- Replacing `ModelInstanceReadinessChecks`/`warm_model_instance`'s public
  signature with a `WarmupRequest`-style API. The auditor confirmed this
  is future OpenSpec Change territory, not required to close this PR.

## Decisions

**Item 5 (`mark_ready` sealing) is intentionally NOT closed by Rust
visibility, and this is a documented trade-off, not an oversight.**
`ModelInstanceManager::mark_ready` is called directly by 13 existing
integration tests in `magnetar-runtime/tests/contract_tests/model_instance.rs`
(a separate compilation unit -- `pub(crate)` would not compile against
it) plus 2 unit tests in `tests.rs`, all using it purely as lifecycle-
testing setup, unrelated to weight materialization. Verified `magnetar-cli`
never calls it (`grep -rn "mark_ready" magnetar-cli/src/` -- no matches),
so restricting visibility would not break the actual embedder in this
repository, but would require editing 15 test call sites for no
behavioral gain: `mark_ready()` takes no `checks` and has no Runtime
context to derive `provider_ready`/`device_ready` from even if hardened,
so hardening it would at best add the same `weights_materialized` check
`warm_model_instance` already gets -- a partial, inconsistent close, not
a real one. The auditor's own reply explicitly authorized this trade-off:
"elle n'est pas nécessairement exigée pour cette PR si elle entraîne une
grosse casse API." Item 4's fix (the `acquire_usage`/`generation_reference`
lifecycle+readiness AND-gate) is the structural mitigation that matters
here regardless: even a `mark_ready()`-forged instance reaches a fully
*consistent* `Ready` state (unlike the `Disabled`-policy bug), so the
lifecycle+readiness gate alone does not catch it -- this residual gap
(an internal/embedder Rust caller reaching into `ModelInstanceManager`
directly, bypassing the intended `warm_model_instance` public contract)
is accepted, documented, and left for a future, narrower follow-up if the
project's threat model changes (e.g. `magnetar-runtime` used by untrusted
third-party embedders rather than the current single first-party CLI).

**`weights_materialized`, `provider_ready`, `device_ready` derivation
lives in `warm_model_instance` (`inference_api.rs`), not in
`ModelInstance`/`ModelInstanceManager` (`model_instance.rs`).**
`model_instance.rs` is generic Core with no access to the Provider
registry or Device list (`ModelInstanceManager` has no `&Runtime`
reference); `warm_model_instance` already takes `runtime: &mut Runtime`
and is the one place with both the instance's own state and Runtime
context to cross-check it against. Alternative considered: derive
`weights_materialized` universally inside `ModelInstance::mark_ready()`
too (closing item 5 as a byproduct) -- rejected per the decision above.

**`weights_materialized` derivation is `caller_value && !resource_bindings.weights.is_empty()`,
not a check against a specific required-weight-name list.** The generic
`ModelInstanceDefinition` only retains an opaque `ModelArtifactId`, not
the manifest's declared tensor list, so "does this exact instance require
these exact N named weights" is not cheaply derivable without new
plumbing threading the manifest/tensor list through to readiness checks.
The audit's own acceptance test (18.3: "actual weight bindings empty,
caller says weights_materialized=true -> Runtime MUST reject") only
requires catching the degenerate case (nothing bound at all), which this
heuristic catches exactly, without new plumbing. A genuinely-weightless
instance (no test fixture in this codebase has one; every manifest used
declares at least one tensor) would be unable to reach `Ready` through
`warm_model_instance` under this heuristic -- an accepted, narrow
limitation, not a real one for this project's current scope.

**`provider_ready`/`device_ready` derivation is trivially `true` when no
Provider/Device is pinned (`placement.provider`/`.device` is `None`).**
Many legitimate instances (including every contract-test fixture) use a
`ResourceAffinity` with no pinned Provider/Device, resolved later at
dispatch time. Treating "nothing pinned yet" as a readiness failure would
reject instances the existing contract already allows to reach `Ready`
generically. When a Provider/Device *is* pinned, it is checked for real
(`providers().provider(binding).and_then(execution_api()).is_some()`;
matching `Device` found in `runtime.devices()` with `availability() ==
Available`).

**`ModelInstance::validate_readiness`'s lifecycle-consistency gate treats
`Warming` as inference-supporting for this purpose (in addition to
`Ready`/`Idle`/`Active`), separately from `ModelInstanceLifecycleState::
supports_inference_use`.** The normal (non-`Disabled`) warmup path
transitions to `Warming` *before* calling `validate_readiness`, so a
successful non-`Disabled` warmup must still be able to compute `Ready`
readiness at that point in order to then transition lifecycle to `Ready`
itself. `supports_inference_use` (used by `acquire_usage`/
`generation_reference`) intentionally does NOT include `Warming` --
warmup itself is not yet a usable state.

## Risks / Trade-offs

- [Risk] The `weights_materialized` heuristic (bindings non-empty) is
  necessary but not individually sufficient proof of *correct*
  materialization (e.g. it would not catch a resource bound under the
  wrong key, or one weight of three genuinely required ones). →
  `WeightMaterializationTransaction` (from `transactional-weight-
  materialization`) is still the actual authority for production
  materialization; this heuristic only closes the specific "nothing was
  ever bound" forgery the audit demonstrated, at the public API boundary
  a caller could otherwise exploit. Full per-weight-name verification
  would need the manifest/tensor-list plumbing noted above; not required
  by the audit's own acceptance criteria.
- [Risk] The `mark_ready()` bypass (item 5) remains real, documented above.
  → Accepted per the auditor's own explicit trade-off allowance; revisit
  if the threat model changes.
