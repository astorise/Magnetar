## Context

A revalidation audit of `runtime-owned-model-instance-readiness-authority`
(commit `9ec9cd3`) corrected a real error in that Change's own design.md:
it had characterized leaving `mark_ready`/`transition_to` publicly
reachable as a trade-off the auditor authorized. The auditor's follow-up
reply (this Change's prompting context) clarified that was a
misreading -- their "DO NOW" list item 5 explicitly required closing the
direct public path to `Ready`, and the "not necessarily required" caveat
was about the *cleanest possible* implementation (splitting `transition_to`
into safe/unsafe primitives), not about leaving the bypass open. This
Change corrects that.

The audit reproduced three concrete, verified gaps (all confirmed by
reading the real code before this Change started, not assumed from the
audit's own description):

- `runtime.model_instances_mut().mark_ready(&id)` -- still `pub`, still
  skips every check.
- `instance_mut(&id).unwrap().transition_to(Ready)` -- an equally valid
  bypass, since `transition_to` was also still `pub`.
- `instance.lifecycle = Ready; instance.readiness = Ready;` -- direct
  field assignment, since both fields were `pub`. This is *not* something
  the audit's own reproduction steps demonstrated explicitly, but is
  structurally implied by both fields being public, and the audit's own
  minimal fix list (item 3) named it directly: "lifecycle/readiness ne
  sont pas directement forgeables via des champs publics mutables."

Given the size of the fix (touching field/method visibility used across
~30 call sites in this crate's own test suite, most in a separate
compilation unit), the user was asked to choose between a narrower
"seal the demonstrated method bypasses only" fix and the full closure
including field privacy; the user chose full closure.

## Goals / Non-Goals

**Goals:** close all three gaps for real (compiler-enforced, not
convention-enforced), migrate every affected test onto the legitimate
public path (`warm_model_instance` with real evidence) rather than
inventing a parallel test-only escape hatch, and strengthen
`weights_materialized`/`provider_ready` derivation per the audit's
remaining two findings (P0-B, P0-C) in the same pass.

**Non-Goals:**
- A `test-support`/`testing` Cargo feature flag exposing the sealed
  primitives for the crate's own tests. Considered and rejected: such a
  feature would be included by `--all-features` (which this project's own
  CI and coverage tooling use), meaning a downstream consumer who also
  builds with `--all-features` would regain the exact bypass this Change
  closes. The chosen alternative -- `pub(crate)` fields/methods, with
  tests migrated to the real public contract -- has no such leak.
- Deriving `kernel_preparation_ready`/`autotuning_ready` from Runtime
  state. Unchanged from round 1's Non-Goals: no generic linkage from a
  `ModelInstanceId` to its expected prepared Kernel set exists in the
  current baseline.
- A `WarmupRequest` redesign of the public warmup API. Still not required;
  this Change closes the forgery paths without touching
  `warm_model_instance`'s or `ModelInstanceReadinessChecks`'s public
  shape.

## Decisions

**`lifecycle`/`readiness` become `pub(crate)`, not fully private.**
Fully private fields would be invisible even to this crate's own test
suite in `tests.rs` (a sibling module of `model_instance`, not a child of
it -- Rust module privacy is not crate-wide). One existing test
(`model_instance_acquire_usage_rejects_ready_readiness_with_incompatible_
lifecycle`, from round 1) deliberately constructs the
`lifecycle=Loading, readiness=Ready` inconsistency directly, to prove the
`acquire_usage`/`generation_reference` safety net holds regardless of how
such a state might arise -- a legitimate, still-necessary defense-in-depth
test that a fully-private field would make impossible to write from
`tests.rs`. `pub(crate)` is exactly the boundary the audit asked for: not
visible or writable by an external embedder's dependency on this crate,
still usable by this crate's own code.

**`ModelInstance::transition_to`/`mark_ready`/`warmup` (and the
`ModelInstanceManager` wrappers) become `pub(crate)`.** Verified before
this decision: `magnetar-cli` (the one real embedder in this repository)
never calls any of these three methods directly (`grep -rn
"mark_ready\|\.transition_to(\|\.warmup(" magnetar-cli/src/` -- no
matches), so this closes nothing a real production caller uses. The only
callers outside `model_instance.rs`/`inference_api.rs` were this crate's
own tests, all migrated (see below).

**Integration tests migrate to `warm_model_instance` with real,
residency-backed evidence -- not a parallel test-only bypass.** For every
test in `magnetar-runtime/tests/contract_tests/model_instance.rs` that
needs a `Ready` instance for reasons unrelated to weight/readiness
semantics (adapter activation, sharing policy, KV cache release, reload,
...), the fix is: `Runtime::initialize(...)` instead of a bare
`ModelInstanceManager::new()`, a real `MemoryManager::allocate` +
`record_tensor_residency` for a fake-but-real weight, then
`warm_model_instance`. This is deliberately the same path a real embedder
must use -- it is proof the contract is actually usable, not a
workaround bolted on to make tests pass. One test
(`provider_and_device_status_drive_instance_lifecycle`) that used to
construct a bare `ModelInstance` and drive it to `Ready` via raw
`transition_to` calls (to test its own reactive `provider_status_changed`/
`device_unavailable` methods) was split into three separately-`Runtime`-backed
instances instead of one instance whose `readiness` field was manually
reset mid-test -- both because raw field reset is no longer available
externally, and because separate instances per scenario is clearer than a
manual reset was to begin with.

**`weights_materialized` also requires a `TensorResidency` record per
bound weight.** `resource_bindings.weights.values().all(|id|
runtime.memory().tensor_residency(id).is_some())`, ANDed with the
existing non-empty check. Closes the audit's test 23.3 (a bare
`TensorResourceId` inserted directly, no residency, must not pass).
Still not a check against a required-weight-name inventory (see round 1's
design.md for why that specific plumbing doesn't exist yet) -- this
closes the "nothing real backs this map entry at all" case, which is what
the audit actually demonstrated as exploitable.

**`provider_ready` also requires
`Provider::status_snapshot().accepts_new_work_by_default()`.** Combines
lifecycle, health, readiness, pressure, and admission -- the Runtime's own
existing authoritative Provider status model (already used elsewhere,
e.g. Provider selection/scheduling), not a new mechanism. Closes the
audit's test 23.5 (a Provider that is registered and offers
`execution_api()` but is `Saturated`/`Draining`/`Unavailable` per its own
status model must not pass).

## Risks / Trade-offs

- [Risk] A future new caller of `ModelInstance`'s public surface could
  still reach `Ready` illegitimately if it is added as `pub` by mistake
  rather than `pub(crate)`, since nothing structurally prevents a new
  method from being added carelessly. → No compiler-enforced mitigation
  beyond code review; documented prominently in the relevant doc comments
  (`ModelInstance::mark_ready`, `::warmup`, `::transition_to`) so the
  intent is visible at the definition site, not just in this design doc.
- [Risk] `weights_materialized`'s residency check can still be satisfied
  by a caller who manually calls the public `MemoryManager::allocate` +
  `record_tensor_residency` for a fake resource, without ever going
  through `WeightMaterializationTransaction` or writing real Provider
  storage. → Accepted, same reasoning as round 1: this raises the bar from
  "insert a map entry" to "perform real Memory Manager admission,"
  meaningfully higher even if not a complete proof of Provider-backed
  storage. Full closure would need the required-weight-inventory plumbing
  noted as out of scope above.
