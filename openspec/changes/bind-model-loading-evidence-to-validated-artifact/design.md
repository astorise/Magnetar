## Context

Verified directly against the code (HEAD `9939232`) before designing
anything, per this session's standing practice of not trusting an audit's
(or a prior round's) characterization without re-checking:

- `LoadedModelContext`/`ModelLoadingResidencyPlan`: every field `pub`, no
  crate-internal constructor. Grep across `magnetar-runtime/tests` and
  `magnetar-cli` for direct field access or struct-literal construction of
  either type: zero hits. The one place that builds a `LoadedModelContext`
  outside `model_loading.rs` itself (`contract_tests/model_instance.rs`'s
  `loaded_context()`) already calls `ModelLoadingCoordinator::load()`, not a
  struct literal. This is a materially smaller blast radius than round 2's
  `lifecycle`/`readiness` sealing (~80 call sites); sealing here is closer
  to zero call sites needing a source change beyond the field-visibility
  line itself.
- `ModelInstanceResourceBindings.weights`/`.memory_allocations`: `pub`, and
  `ModelInstanceDefinition.resource_bindings` (the field holding it) is also
  `pub`. Both matter: sealing only the inner fields would still let an
  external caller replace the *whole* `resource_bindings` value (a plain
  field assignment, not a struct literal -- `Clone` doesn't require field
  visibility) with one cloned from a different, already-Ready instance,
  which is exactly the audit's "Artifact mismatch" reuse scenario. Both the
  outer field and the two inner fields need `pub(crate)`.
- One real caller mutates these fields directly today:
  `contract_tests/model_instance.rs`'s `bind_fake_weight` -- and it already
  performs a real `write_tensor` + real `record_tensor_residency` first
  (hardened in round 5), then pokes `resource_bindings.weights`/
  `memory_allocations` by hand instead of going through
  `WeightMaterializationTransaction::commit`. This is precisely the "some
  bytes exist in Provider storage" vs. "this instance's own authorized
  transaction produced this binding" gap the audit describes in the
  abstract (sections 12-14) demonstrated concretely by this crate's own
  test helper.
- The production weight-materialization path already has the right shape
  and is already the *only* path production code uses:
  `bind_qwen_fixture_weights` -> `materialize_model_instance_weights` ->
  `WeightMaterializationTransaction::{begin, stage_weight, commit}`. All of
  these are private free functions/a private struct today (not even
  `pub(crate)` in the sense of being usable from `tests/contract_tests` --
  that directory compiles as a separate crate). `commit` already does
  admission-then-write-then-residency in the correct order and already
  calls `mark_ready` directly (same-crate `pub(crate)` call, not through
  `warm_model_instance`) once every staged weight succeeds -- this is fine:
  it is not caller-reachable from outside the crate, and it *is* the
  transaction whose successful completion readiness should trust.
- No per-tensor or per-artifact content digest exists anywhere generic.
  `ModelTensorMetadata` (`model.rs`) has no digest field. The one digest
  check in this codebase (`bind_qwen_fixture_weights` comparing
  `e2e_fixture_weight_digest(&fixture.weights)` against a hardcoded
  `E2E_FIXTURE_WEIGHT_DIGEST` constant) is specific to one E2E fixture, not
  a manifest-declared, artifact-source-agnostic check any caller gets for
  free. Building that generically is real, valuable work, but it changes
  the `model-artifact`/`model-loading` manifest schema, not just
  `model-instance` readiness -- a different capability boundary than this
  Change, and explicitly out of scope here (see Non-Goals).

## Goals / Non-Goals

**Goals:**
- `LoadedModelContext`/`ModelLoadingResidencyPlan` become impossible to
  construct outside `ModelLoadingCoordinator::load()`.
- Weight resource bindings (`resource_bindings.weights`/
  `memory_allocations`) become impossible to set outside the one authorized
  materialization transaction.
- `weights_materialized` readiness is derived from Runtime-issued evidence
  that (a) exists for this specific instance, (b) matches this instance's
  own declared `ModelArtifactId`, and (c) matches the currently-bound
  resource-id set -- not from probing whatever bytes currently happen to sit
  in Provider storage.
- A materialization-evidence record from instance A's transaction cannot
  make instance B (a different instance, whether for the same or a
  different artifact) appear materialized.
- `derive_effective_readiness_checks` no longer depends on
  `ProviderExecutionApi::read_tensor`/`HostTensor` at all, closing the
  companion P1 as a consequence of the redesign rather than a separate fix.
- The one legitimate external-facing way to materialize weights
  (`contract_tests` and any real embedder) is the same transaction
  production code uses -- no parallel test-only bypass.

**Non-Goals:**
- **Byte-content provenance.** This Change does not verify that
  materialized bytes are bit-identical to the specific validated Model
  Artifact's declared tensor content. It closes "who is authorized to
  produce a binding" (only the one transaction, for this one instance), not
  "are these the *correct* bytes for that artifact." The existing
  fixture-specific digest check in `bind_qwen_fixture_weights` is untouched
  and keeps doing its job for that one call site. Generalizing content
  verification to every artifact source needs a manifest-level digest field
  threaded from artifact parsing through `LoadedModelContext` (the same
  shape of plumbing `required_weight_names` already established), which
  touches the `model-artifact`/`model-loading` schema -- proposed as a
  follow-up Change, not folded in here.
- **Option 2 (opaque `LoadedModelHandle`/registry replacing
  `create_model_instance(&LoadedModelContext)`'s signature).** Considered
  and rejected for this Change: the grep above found zero external
  dependents on `LoadedModelContext`'s public fields or direct
  construction, so `pub(crate)` sealing already closes the forgery vector
  Option 2 would also close, at a fraction of the blast radius (no
  signature change to `create_model_instance`, `from_loaded_context`, or
  any of their many production/fixture call sites). Revisit only if a
  future audit finds a forgery path `pub(crate)` sealing does not cover.
- Re-litigating `mark_ready`/`transition_to`/`warmup` visibility or the
  `lifecycle`/`readiness` field sealing -- both already closed by prior
  rounds and reconfirmed still closed by this audit round.
- A `WeightMaterializationState` state machine
  (`NotStarted`/`InProgress`/`Complete`/`Failed`) -- superseded by the
  evidence-record design below, which gets the same anti-forgery property
  without a publicly-observable-but-not-forgeable state enum.

## Decisions

**Weight materialization evidence is a Runtime-owned, per-instance record
minted only by `WeightMaterializationTransaction::commit`, not a new public
field on `ModelInstance` or `ModelInstanceDefinition`.** Considered adding a
`materialization_generation: u64` or similar public/`pub(crate)` field
directly on the instance: rejected for the same reason round 5 rejected a
public `WeightMaterializationState` flag -- anything living on the instance
itself, reachable through `Runtime::model_instances_mut()`, is either
publicly settable (forgeable) or `pub(crate)`-settable (works, but is
indistinguishable in shape from the field-sealing this Change already does
for `resource_bindings`, so keeping evidence in a separate Runtime-owned
table -- keyed by `ModelInstanceId`, populated only by `commit`, cleared only
by `unload_model_instance`/transaction `abort` -- keeps "what changed a
binding" and "what proved a binding legitimate" as two independently-sealed
surfaces rather than one, so a future bug in one does not automatically
compromise the other.

**Evidence records the committed resource-id set, not just a boolean or a
counter.** `derive_effective_readiness_checks` compares this set against
`resource_bindings.weights`'s *current* values exactly (not merely "an
evidence record exists"). Because only `commit` ever writes both
`resource_bindings.weights` and the evidence record, and it writes them
together, the two cannot drift apart through this crate's own code today --
but keeping the check explicit rather than relying on that invariant holding
forever is cheap and matches this session's established preference for
checks that fail closed even if a future internal change reintroduces
drift, rather than checks that are only correct because nothing currently
violates an unstated assumption.

**Evidence is looked up by `ModelInstanceId` and compared against that same
instance's own `definition.artifact`, not stored keyed by
`ModelArtifactId`.** This is what makes the audit's "Artifact mismatch"
scenario fail correctly: instance B's evidence lookup only ever finds
evidence `commit` minted for B's own id (from B's own transaction); nothing
about instance A's artifact, transaction, or evidence is reachable from B's
lookup key at all, regardless of whether A and B share the same
`ModelArtifactId` or not. The additional `artifact` match against
`definition.artifact` is defense-in-depth for the case where a future change
allows an instance's `definition.artifact` to be reassigned after creation
(not possible today -- `artifact` has no setter -- but the check is cheap
and matches the same "fail closed even against future drift" reasoning
above).

**The public materialization entrypoint takes `&BTreeMap<String,
HostTensor>` (unchanged from the existing private
`materialize_model_instance_weights` signature), not a new caller-supplied
evidence/proof type.** The transaction's *successful completion* is the
proof; there is no separate token for a caller to construct, forge, or
mismanage. This keeps the API surface a caller must learn identical to what
`bind_qwen_fixture_weights` already calls internally today.

**`resource_bindings` (the whole field) becomes `pub(crate)`, not just its
two inner fields.** See Context: sealing only `weights`/`memory_allocations`
leaves whole-field replacement (`instance.definition.resource_bindings =
other.resource_bindings.clone()`) open, since `Clone` doesn't require field
visibility. `released_memory_allocations`/`released_provider_resources` (the
struct's other two fields, used by the release path) become `pub(crate)`
alongside `weights`/`memory_allocations` for the same reason, even though
the audit did not specifically call them out -- leaving them `pub` while
their siblings become `pub(crate)` would not itself be a materialization
forgery vector (they only ever narrow what counts as "still resident"), but
a mixed-visibility struct where only the fields a specific audit
enumerated are sealed is exactly the kind of narrow, symptom-scoped fix
this session's retrospectives (round 1 -> round 2) have already shown
under-closes the general property. Read-only public accessors are added if
and when `cargo build --tests` shows a real external caller needs one
(none does today per the Context grep).

## Risks / Trade-offs

- [Risk] Byte-content provenance remains unverified generically (see
  Non-Goals) -- a caller with legitimate access to the public
  materialization entrypoint can still supply the *wrong* (but not
  externally-forged-by-a-different-path) bytes for a given artifact, and
  evidence will legitimately mark the instance materialized. → Accepted for
  this Change: this is a strictly smaller residual than today's baseline
  (today, a caller doesn't even need the public entrypoint -- direct field
  access works), and is explicitly named as this Change's boundary with a
  concrete named follow-up rather than silently left unaddressed.
- [Risk] Promoting `materialize_model_instance_weights` to a public `Runtime`
  method changes today's "everything in `first_native_runtime.rs` beyond a
  handful of `pub fn`s is private" posture for one more function. →
  Accepted: it is the one function this Change's whole design routes every
  legitimate caller through; keeping it private while adding a second
  public function that just calls it would add indirection with no
  additional safety.
- [Risk] `pub(crate)`-sealing `ModelInstanceResourceBindings`'s fields
  removes `contract_tests`' ability to directly assert on bound weight
  names/resource ids after `warm_model_instance` succeeds (today's
  `bind_fake_weight`/assertions may read these fields for verification, not
  only write them). → Mitigation: add a narrow public read-only accessor
  (e.g. `ModelInstanceDefinition::bound_weight(name: &str) -> Option<&
  TensorResourceId>`) if `cargo build --tests` shows a real read dependency;
  none is known yet from the Context grep, which only found write sites.

## Migration Plan

Single-PR change, no runtime data migration (in-memory Runtime state only,
no persisted format). Implementation order, each step buildable/testable
before the next (matching this session's established practice of never
landing a half-migrated visibility change):

1. Add the `MaterializationEvidence` record type and Runtime-owned storage
   (empty until step 3 populates it) -- additive, no visibility changes yet.
2. Seal `LoadedModelContext`/`ModelLoadingResidencyPlan` fields to
   `pub(crate)`; fix any resulting build errors (expected: none outside
   `model_loading.rs` per the Context grep).
3. Promote `materialize_model_instance_weights` to a public `Runtime`
   method; have `WeightMaterializationTransaction::commit` mint/replace the
   instance's evidence record alongside its existing binding writes.
4. Seal `ModelInstanceDefinition.resource_bindings` and
   `ModelInstanceResourceBindings`'s four fields to `pub(crate)`; migrate
   `contract_tests/model_instance.rs`'s `bind_fake_weight` onto the new
   public entrypoint from step 3; fix any other resulting build errors.
5. Switch `derive_effective_readiness_checks`'s `weights_materialized` from
   `read_tensor` probing to evidence matching; remove the now-dead
   `read_tensor`-based check.
6. Clear an instance's evidence record in `unload_model_instance` and in
   `WeightMaterializationTransaction::abort`'s failure path (mirroring how
   `TensorResidency` cleanup already works, per
   `invalidate-tensor-residency-on-release`).
7. New tests per proposal.md's Impact section; full workspace suite,
   `cargo doc`, wasm32 check, coverage ratchet, `openspec validate --all
   --strict`, live `qwen-test`, full CI on the exact final HEAD.

Rollback: revert the commit(s); no persisted state to migrate back.

## Open Questions

- Exact name/shape of the read-only accessor(s) added under Risk 3 above,
  if `cargo build --tests` shows one is actually needed -- deferred to
  implementation time rather than guessed here.
- Whether the follow-up "verify materialized bytes against a manifest-
  declared digest" Change is worth doing before or after v0.1 ships, given
  it is `model-artifact`/`model-loading` schema work, not `model-instance`
  work -- left for the user/next audit round to prioritize, not decided
  here.
