## Context

A fifth audit round (commit `dd280df`) confirmed every gap from round 2
(`seal-model-instance-readiness-authority`) was genuinely closed --
public `mark_ready`/`transition_to`/`warmup`, public mutable
`lifecycle`/`readiness` fields, Provider readiness based only on
`execution_api()`, and bare weight bindings without residency were all
verified fixed by re-reading the current code before accepting the
audit's findings, matching this session's standing practice. It then
found two further real gaps, both verified against the actual code and
the actual canonical spec text before this Change started:

- `resume_model_instance` -> `ModelInstance::resume()`, confirmed still
  performing `Suspended -> Loading -> Ready` via two internal
  `transition_to` calls with no call to `derive_effective_readiness_checks`
  anywhere in between.
- `derive_effective_readiness_checks`'s `weights_materialized`, confirmed
  to check only `!bindings.is_empty() && bindings.values().all(|id|
  tensor_residency(id).is_some())` -- no inventory-completeness check, no
  check that the residency's claimed Provider actually holds the tensor.
  Confirmed the canonical spec already requires both: `model-loading`'s
  "Qwen Loading Validates Tensor Inventory" ("Model Loading SHALL use
  Qwen Component tensor inventory metadata to validate required tensors
  before ready Model Instance publication") and "Partial Loading Policy"
  ("Partial loading SHALL...NOT produce ready state if required parts are
  missing") both predate this Change.

Given the second gap's fix requires new field plumbing (not just
tightening an existing check), the user was asked whether to fix both
gaps now or defer the harder one (inventory completeness); the user chose
to do both.

## Goals / Non-Goals

**Goals:** close both gaps for real, verified by direct code
re-inspection rather than trusting either the audit's or my own prior
characterization, without changing `create_model_instance`'s or
`ModelInstanceDefinition::from_loaded_context`'s public signatures
(avoiding the kind of large cascading change `seal-model-instance-
readiness-authority` needed for field/method visibility).

**Non-Goals:**
- A `WeightMaterializationState` state machine
  (`NotStarted`/`InProgress`/`Complete`/`Failed`) as the audit's own
  "Option recommandée" suggested. Considered: this would need either (a)
  a `pub(crate)`-only setter only the real transaction can call --
  reintroducing the exact "the integration test crate can't legitimately
  construct evidence" problem `seal-model-instance-readiness-authority`
  already solved by migrating tests onto the real public contract, or (b)
  a publicly-settable flag, which would just move the forgery surface
  rather than closing it. The chosen alternative -- checking the
  Provider's own real storage via `read_tensor`, which is not
  forgeable without actually writing real bytes to it -- achieves the
  same property without new Runtime-owned state.
- Full manifest/architecture validation beyond tensor *names* (dtype,
  shape, quantization compatibility, etc.). Out of scope: the audit's own
  required tests (23.1-like: "Inventaire incomplet") only ask for
  presence-by-name, matching what `model-loading`'s existing spec
  requirements already state.

## Decisions

**`weights_materialized`'s Provider-backing check reads real Provider
storage (`ProviderExecutionApi::read_tensor`) instead of adding new
Runtime-owned bookkeeping.** A `TensorResidency` is a plain struct with
entirely public fields (`pub tensor`, `pub allocation`, `pub placement`,
`pub affinity`, ...) -- any new field added to it, or any new sibling
Runtime-owned table gated behind a `pub(crate)` setter, remains either
publicly constructible (not a real proof) or crate-internal-only
(unusable by `contract_tests`, which is an external compilation unit,
without reintroducing a parallel test-only bypass). The one thing a
caller genuinely cannot fabricate without doing the real thing is the
Provider's own storage: resolving the residency's recorded Provider and
calling `read_tensor` confirms bytes were actually written, which is
exactly what a forged residency (Memory Manager allocation + a manually
constructed `TensorResidency`, no `write_tensor` call) cannot produce.

**`required_weight_names` is threaded through `LoadedModelContext` (a new
field populated at `ModelLoadingCoordinator::load()` time from
`manifest.tensors`), not through a new `create_model_instance` parameter.**
`create_model_instance` has no direct manifest access and is called
pervasively (first-native production code, most of this crate's own test
fixtures); widening its signature would cascade far beyond this Change's
actual scope. `load()` already receives `&ModelManifest`, and
`ModelInstanceDefinition::from_loaded_context` already receives
`&LoadedModelContext` -- carrying the tensor names through that existing
path needed no signature changes at either the `create_model_instance` or
`from_loaded_context` call sites, only a new struct field populated once,
at the one place (`load()`) that already has the manifest in scope.

**`required_weight_names` on `ModelInstanceDefinition` is `pub(crate)`,
not `pub`, and empty means "unknown" (falls back to the prior
presence-only heuristic), not "nothing required."** A `pub` field would
let a caller redeclare it after creation to make an incomplete binding
set appear complete -- the same class of forgery this Change closes for
weight bindings themselves. Treating empty as "unknown" rather than
"trivially satisfied" matters for every generically-constructed
`ModelInstance` in this crate's own test suite that has no loaded
manifest behind it at all (most of `tests.rs`'s and `contract_tests`'s
fixtures) -- those must keep working exactly as before, since the
Runtime genuinely does not know what they require.

**`ModelInstance::resume()` only reaches `Loading`; `resume_model_instance`
completes the transition by calling `warm_model_instance` (the same
function `create`-then-warm and `reload`-then-warm callers already use),
not a bespoke resume-specific readiness path.** This reuses the exact
same Runtime-derived-evidence machinery rather than duplicating it, and
matches the audit's own "Variante A" recommendation. `ModelInstance::resume`
stays `pub` (unlike `mark_ready`/`transition_to`/`warmup`): since it no
longer reaches `Ready` by itself, calling it directly just leaves the
instance in `Loading` -- harmless, not a forgery vector, so it does not
need the same sealing.

## Risks / Trade-offs

- [Risk] `read_tensor`-based Provider verification only proves *a*
  Provider currently holds bytes for that resource ID, not that those
  bytes are the *correct* ones for the declared tensor (dtype, shape,
  content). → Accepted: proving content correctness would need comparing
  against the manifest's declared tensor metadata (shape/dtype), which
  the existing `WeightMaterializationTransaction` production path already
  does at write time; this Change's concern is specifically the forgery
  the audit demonstrated (a residency claiming a Provider write that
  never happened), which `read_tensor` returning `Some` directly refutes.
- [Risk] A `ModelInstance` whose manifest declared zero tensors (a
  degenerate case; none exist in this codebase's fixtures) would have
  `required_weight_names` empty and fall back to the presence-only
  heuristic, identical to today's behavior. → Accepted as consistent with
  "empty means unknown," not a regression from the current baseline.
