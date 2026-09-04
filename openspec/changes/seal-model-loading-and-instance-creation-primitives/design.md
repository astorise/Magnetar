## Context

Round 10 sealed trust at the `inference_api::load_model`/`load_model_observed`
facade and provenance at `Runtime::create_model_instance`. Round 11 found
both seals sit one layer above the primitives that actually do the work,
and those primitives stayed `pub`:

- `ModelLoadingCoordinator::load(&mut self, request, manifest, trust:
  &ModelTrustDecision, memory: &mut MemoryManager)` -- callable directly
  with a self-built `ModelTrustStore`'s (legitimately evaluated, not
  forged) decision and a self-built `MemoryManager`, never touching
  `Runtime` at all.
- `ModelInstanceDefinition::from_loaded_context` + `ModelInstanceManager::
  create` -- callable directly with caller-chosen architecture/affinity,
  reachable via `Runtime::model_instances_mut()`, never touching
  `Runtime::create_model_instance`'s cross-checks.

Both are demonstrated, not hypothetical: `magnetar-runtime/tests/
contract_tests/` -- built and run as a separate crate against
`magnetar_runtime`'s public API, the same vantage point a real embedder
has -- already uses both today. Inspection before writing this design
confirmed neither primitive is used by any non-test, non-test-only code
(`magnetar-cli`, `formats/gguf`, `formats/safetensors`): sealing them is
safe for every real caller, only test code needs to move.

Round 11 also found a real self-contradiction: round 10's own new
`model-instance` requirement blessed an unresolved plan field's
caller-supplied value as "a legitimate choice ... SHALL NOT be
constrained," while the pre-existing `inference-api` requirement says
"Runtime SHALL own Provider and Device selection." `ModelInstancePlacement::
new` does, in fact, copy the caller's affinity straight into the
effective placement whenever the plan resolved nothing -- there is no
Runtime-side arbitration step at instance-creation time. The existing
`ResolutionPolicy`/`BuiltInResolutionPolicy` mechanism resolves
Capability/Provider candidates at *execution* time (kernel dispatch); it
has no integration point for instance-creation-time placement today, and
building one is real, separately-scoped design work, not a provenance
seal. The user chose (via AskUserQuestion) to correct the spec text to
honestly describe today's behavior as a documented limitation rather than
build that mechanism now.

## Goals / Non-Goals

**Goals:**
- Neither `ModelLoadingCoordinator::load` nor the
  `ModelInstanceDefinition`/`ModelInstanceManager::create` pair is
  reachable from outside the crate by any path that does not go through
  `Runtime`'s sealed trust and cross-checks.
- Every test that is *actually* testing one of these primitives' own
  contract (not merely using it as a convenient way to reach some other
  state) keeps testing it, just from inside the crate where `pub(crate)`
  access still works -- no coverage is silently dropped.
- Every test that is testing genuinely external-consumer-observable
  behavior (Model Instance lifecycle, KV cache interaction, sharing,
  warmup, unload) migrates to the real public entrypoint
  (`Runtime::create_model_instance`) it should have been using anyway.
- The `model-instance`/`inference-api` spec contradiction is resolved by
  making both texts describe one true thing, not by making the code do
  something new.

**Non-Goals:**
- Building Runtime-side Provider/Device resolution for instance creation
  (explicitly deferred by the user's choice; tracked as a documented
  limitation, the same pattern as the existing F32-only digest coverage
  note).
- Changing `ModelLoadingCoordinator`'s or `ModelInstanceManager`'s
  internal behavior in any way beyond visibility -- this Change is a seal,
  not a redesign of loading or instance-creation semantics.
- Touching `ModelTrustStore`'s or `ResourceAffinity`'s own public API --
  both remain legitimately public value types; what changes is which
  *sinks* (`load`, `create`) are willing to accept caller-supplied values
  as authoritative.

## Decisions

### D1: Seal by narrowing visibility in place, not by introducing wrapper types

`pub fn load(...)` becomes `pub(crate) fn load(...)`; same for
`from_loaded_context` and `create`. No new type or trait is introduced to
gate access -- the existing `pub(crate)` convention this crate already
uses extensively (every `ModelInstanceDefinition` field, `ModelTrustDecision::
new`, `ModelInstanceDefinition.artifact`/`.architecture`) is the
established, minimal-blast-radius tool for exactly this problem, and
introducing a different mechanism here would be inconsistent without
adding anything.

Alternative considered: keep `load`/`create` public but require an
opaque, crate-only-constructible "authority token" parameter. Rejected --
this is strictly more machinery than `pub(crate)` for the same effect,
and every prior round's equivalent fix (round 8's `resource_bindings`,
round 9's `artifact`/`architecture`) used plain visibility narrowing.

### D2: Relocate tests by what they actually test, not wholesale

Not every `contract_tests` site that happens to call `.create()` or
`.load()` is testing *that primitive's own contract* -- most are using it
as the only way, before this Change, to get a ready `ModelInstanceId`/
`LoadedModelContext` so they can test something else (KV cache release on
unload, sharing policy, warmup checks). Those migrate to
`Runtime::create_model_instance`/`inference_api::load_model` in place,
staying in `contract_tests`, because that is the real public path an
embedder uses and these tests gain fidelity by using it too.

A small minority genuinely test the primitive's own contract and cannot
be expressed through the higher-level API at all:
- `model_loading.rs`'s untrusted-rejection, memory-budget/quantization/
  allocation-failure-mapping, and Ready-context-shape tests inspect
  `ModelLoadingCoordinator`/`LoadedModelContext` details (`context.plan()`,
  `coordinator.observations()`, exact `ModelLoadingErrorCode`s) that
  `Runtime`'s wrapper does not re-expose 1:1, and deliberately probe
  loading in isolation from any `Runtime`/`MemoryManager` a full instance
  creation would require.
- `model_instance.rs`'s `cloned_definition_does_not_inherit_weight_authority`
  and `reload_replacement_does_not_inherit_original_weight_authority`
  specifically prove `ModelInstanceManager::create` resets
  `resource_bindings` regardless of what a hand-cloned definition
  carried -- the test's entire point is calling `.create()` directly on a
  definition obtained by cloning, which has no `Runtime::
  create_model_instance` equivalent (that method never accepts an
  existing `ModelInstanceDefinition` at all).

These move into `magnetar-runtime/src/tests.rs`, the crate's own internal
unit-test file (already home to this session's other `pub(crate)`-only
regression tests), not a new sibling file -- there is no precedent in
this crate for a per-module internal test file outside
`first_native_runtime`'s (which exists specifically to keep ~1350 lines
of test source out of coverage measurement; six-to-eight relocated tests
here do not justify the same treatment).

### D3: Spec correction, not new requirement removal

`model-instance`'s "Model Instance References Architecture Implementation"
requirement keeps its architecture/plan and resolved-affinity/plan
cross-check language (round 10, unchanged, still correctly enforced) but
loses the "SHALL NOT be constrained" framing for the unresolved case,
replaced with an explicit statement that this is a known gap: the
caller's value becomes effective placement directly, Runtime performs no
arbitration. `inference-api`'s "Provider Preferences Are Non-Authoritative"
requirement is not weakened -- its SHALL text is unchanged -- but gains a
cross-reference noting the one place today's implementation does not yet
fully satisfy it, so a future reader (or audit) sees one coherent account
instead of two requirements that read as flatly contradictory.

## Risks / Trade-offs

- **[Risk]** Relocating tests out of `contract_tests` slightly reduces
  that suite's role as "prove the public API alone is sufficient" for the
  relocated cases. → Accepted: those specific cases were never truly
  testing the public API to begin with (they used a bypass that no longer
  exists); the *replacement* tests for the same states now go through the
  real public API, which is a net fidelity improvement for the tests that
  remain in `contract_tests`.
- **[Risk]** The Provider/Device spec correction documents a real gap
  (caller preference becomes authoritative placement) without closing
  it. → Accepted per explicit user decision; tracked in CHANGELOG's
  Known Limitations alongside the existing F32-only digest note, in the
  same style, so it is not lost.

## Migration Plan

1. Confirm (already done) no non-test caller uses either primitive.
2. Seal both primitives.
3. `cargo build`/`cargo test` enumerate every break; relocate per D2.
4. Add external-bypass regression tests proving the seal from
   crate-internal code (the closest a same-crate test can get to
   demonstrating "no external crate can compile this anymore").
5. Correct the two spec texts per D3; add the new `model-loading` and
   `model-instance` reachability requirements.
6. Full verification suite; push; CI; archive; `CHANGELOG.md`.

No rollback strategy beyond git revert; pre-release branch, no deployed
consumers.

## Open Questions

None -- the one real design fork (spec correction vs. new resolution
mechanism) was resolved by the user before this design was written.
