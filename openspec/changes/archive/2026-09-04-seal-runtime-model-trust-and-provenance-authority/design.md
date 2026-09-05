## Context

Round 9 sealed the *direct* forgery paths for Model Instance authority:
`ModelTrustDecision::new` and `ModelInstanceDefinition.artifact`/`.architecture`
became `pub(crate)`. Round 10 found that each seal has a sibling gap one
level up, reachable through code that is still fully `pub`:

1. **Trust.** `ModelTrustStore` (`model.rs:775`) has every field `pub` and a
   fully public `.trust_digest()`/`.trust_source()`/`.evaluate()` API. Any
   code with library access can build a store that trusts exactly the
   artifact it wants and get a real (not forged) `Trusted` decision back.
   Sealing the decision constructor didn't touch this, because the decision
   really is legitimately derived -- from a store the caller built to order.

2. **Instance provenance.** `Runtime::create_model_instance(loaded,
   architecture, affinity)` never compares `architecture`/`affinity`
   against `loaded.plan()`, even though `ModelLoadingResidencyPlan`
   (`model_loading.rs:266`) already carries a resolved `architecture:
   ModelArchitecture` and, when the loading phase settled on one, a
   `provider_binding`/`device_binding`.

3. **Weight provenance.** `stage_weight` only ever checks a per-tensor
   content digest, and digests only exist for `F32` tensors (the only
   dtype `host_tensors_from_artifact_bytes` can materialize into
   `HostTensor` at all). A tensor the manifest declares quantized has no
   digest, so nothing stops a caller from handing `stage_weight` a
   fabricated `F32` `HostTensor` under that tensor's name.

Investigating (1) before design started surfaced a structural fact that
changes its scope: `load_model`/`load_model_observed` (`inference_api.rs:
349`) are free functions taking a caller-owned `&mut ModelLoadingCoordinator`
and `&mut MemoryManager`. `Runtime` does not hold a `ModelLoadingCoordinator`
anywhere in the codebase today -- confirmed by grep across `runtime.rs` and
every real call site (`magnetar-cli/src/commands.rs:713`, the qwen-test
live E2E fixture and two other sites in `first_native_runtime.rs`, and
three sites in `tests.rs`). Every one of those builds its own coordinator
and memory manager, calls `load_model`, and only afterward hands the
resulting `LoadedModelContext` to a separate `Runtime` via
`create_model_instance`. A trust seal that only touches `ModelTrustStore`
would still leave every real call site free to fabricate a decision from a
self-built store, because nothing forces the store used for evaluation to
be the same one a deployment authority configured. Closing (1) for real
means coupling `load_model` to a `Runtime`-owned, once-configured trust
policy, not narrowing `ModelTrustStore`'s visibility.

## Goals / Non-Goals

**Goals:**
- A `Runtime` instance is the sole source of the trust decision used by any
  `load_model` call that goes through it; that decision is not
  caller-suppliable per call.
- `create_model_instance` rejects `architecture`/`affinity` that disagree
  with what the loading phase already resolved, without forbidding
  legitimate choices the plan never resolved (`kind`, `required_capabilities`,
  and any plan field left `None`).
- `stage_weight` rejects materialized content whose shape or declared
  dtype disagrees with the manifest, independent of whether a content
  digest happens to exist for that tensor.
- Every existing real call site (CLI, qwen-test live fixture, both test
  modules) is migrated, not left broken or duplicated behind a legacy path.

**Non-Goals:**
- Redesigning `ModelTrustStore`'s own matching semantics (digest/source/
  publisher trust lists) -- it stays exactly as expressive as it is today;
  only *who gets to choose which store is authoritative for a given load*
  changes.
- Making `ModelLoadingCoordinator` itself `Runtime`-owned as a general
  matter beyond what trust-sealing requires -- `Runtime` gains what it
  needs to own trust evaluation for loads that go through it; it does not
  absorb every field of `ModelLoadingCoordinator`.
- Widening digest coverage to non-F32 dtypes (still tracked as the
  existing, separately-scoped "F32-only by materialization capability"
  limitation) -- the shape/dtype check added here is a *cheaper, coarser*
  guard than a digest, not a replacement roadmap item for real quantized
  digests.
- Changing `ModelArchitectureImplementation.kind`/`.required_capabilities`
  semantics -- confirmed to have no plan counterpart, so they stay
  caller-chosen.

## Decisions

### D1: `RuntimeBuilder::trust_store(ModelTrustStore)`, `load_model` takes `&Runtime`

`RuntimeBuilder` gains `.trust_store(ModelTrustStore) -> Self` (default:
`ModelTrustStore::default()`, matching today's CLI behavior of an empty
deny-by-default store). `Runtime` stores it in a private field with no
public getter that returns an owned/mutable copy -- only
`pub(crate) fn trust_store(&self) -> &ModelTrustStore` for internal use.
`load_model`/`load_model_observed` drop the `trust: &ModelTrustDecision`
parameter and instead take `runtime: &Runtime`, evaluating
`runtime.trust_store().evaluate(manifest)` internally.

Alternative considered: give `Runtime` a `set_trust_store` *after*
`build()`, so it can be reconfigured post-construction. Rejected -- that
reopens exactly the gap this closes (code downstream of construction could
still swap in a self-serving store). Configuration happens once, at
`RuntimeBuilder`, before the `Runtime` a caller receives can run any load.

Alternative considered: make `load_model` a method on `Runtime` instead of
a free function taking `&Runtime`. Rejected for this Change -- `Runtime`
does not otherwise own `ModelLoadingCoordinator`/`MemoryManager` state
(loading is deliberately decoupled from live Runtime residency until a
`ModelInstance` is created), and folding that ownership in is a larger,
separately-scoped change (`Non-Goals`). Taking `&Runtime` alongside the
existing `coordinator`/`memory` parameters is the minimal coupling that
still makes trust Runtime-sourced.

### D2: `create_model_instance` cross-checks against `loaded.plan()`

Add a `pub(crate) fn architecture(&self) -> &ModelArchitecture` accessor is
unnecessary -- `plan().architecture` is already `pub` reachable through
`ModelLoadingResidencyPlan`'s existing `architecture` field (confirmed
`pub(crate)` with the plan itself exposed read-only via `LoadedModelContext
::plan()`). `create_model_instance` gains, before constructing the
definition:

```
if architecture.architecture != loaded.plan().architecture {
    return Err(ModelInstanceError::ArchitectureMismatch { .. });
}
if let Some(expected_provider) = loaded.plan().provider_binding.as_ref()
    && affinity.provider() != Some(expected_provider) {
    return Err(ModelInstanceError::AffinityMismatch { .. });
}
if let Some(expected_device) = loaded.plan().device_binding.as_ref()
    && affinity.device() != Some(expected_device) {
    return Err(ModelInstanceError::AffinityMismatch { .. });
}
```

Two new `ModelInstanceError` variants (`ArchitectureMismatch`,
`AffinityMismatch`), following the existing struct-variant-with-`reason`-or
typed-fields convention already used for `InvalidLifecycleTransition`.

Alternative considered: a single `ProvenanceMismatch { reason: String }`
catch-all. Rejected -- this crate's established convention (round 8/9) is
specific, typed error variants over string-reason catch-alls wherever the
mismatched values are structured data available at the call site; a caller
can match on the specific variant instead of parsing a reason string.

### D3: `stage_weight` shape/dtype check precedes the digest check

`LoadedModelContext`/`ModelInstanceDefinition` gain
`required_weight_shapes: BTreeMap<String, (Vec<u64>, ModelDType)>`,
populated in `ModelLoadingCoordinator::load()` from `manifest.tensors`
exactly like `required_weight_digests` already is (same commit pattern,
same file, same loop). In `stage_weight`, before the existing digest
check:

```
if let Some((expected_shape, expected_dtype)) = required_weight_shapes.get(name) {
    if &tensor.shape != expected_shape || *expected_dtype != ModelDType::F32 {
        return Err(InferenceApiError::WeightShapeOrDtypeMismatch { reason: .. });
    }
}
```

A tensor with no entry in `required_weight_shapes` (name absent from the
manifest) is unaffected by this check -- that is already caught elsewhere
(`required_weight_names`), and this Change does not touch that path.

Alternative considered: reuse `WeightContentDigestMismatch` for this too.
Rejected -- shape/dtype mismatch and digest mismatch are diagnostically
distinct failures (the former says "this could never be the right tensor
at all," the latter says "this might be the right tensor with wrong
bytes"); collapsing them loses information a caller debugging a bad
integration would want.

### D4: Migration order for the seven `load_model` call sites

Each site currently does `ModelTrustStore::default().evaluate(&manifest)`
(or an equivalent local `trusted()`/`untrusted()` test helper) and passes
the result inline. Each migrates to building/reusing a `Runtime` configured
with the equivalent `ModelTrustStore` via `RuntimeBuilder::trust_store(..)`,
then calling `load_model(&mut coordinator, &mut memory, request, &manifest,
&runtime)`. Test helpers that need an *untrusted* decision configure a
`Runtime` whose trust store trusts a different digest than the manifest
under test declares (mirrors what `evaluate` already does today, just
sourced from the `Runtime` instead of an inline store).

## Risks / Trade-offs

- **[Risk]** Breaking change to `load_model`'s public signature affects any
  out-of-tree embedder. → Documented as **BREAKING** in the proposal and
  `CHANGELOG.md`; this crate is pre-1.0 and Architecture Freeze gates are
  explicitly about catching exactly this class of gap before it goes public.
- **[Risk]** Coupling `load_model` to `&Runtime` without folding in
  `ModelLoadingCoordinator`/`MemoryManager` ownership leaves an asymmetry
  (trust is Runtime-sourced, coordinator/memory are still caller-owned)
  that a future audit could flag as inconsistent. → Accepted as an explicit
  Non-Goal; folding those in is real, separately-scoped work this Change
  does not need in order to close the trust gap specifically.
- **[Risk]** `provider_binding`/`device_binding` on the plan may be `None`
  far more often than populated in today's fixture/test paths, making the
  affinity cross-check exercise the "no constraint" branch almost
  exclusively in existing tests. → Mitigated by writing a dedicated test
  that forces a plan with a resolved `provider_binding`/`device_binding`
  and asserts the mismatch path actually rejects.

## Migration Plan

1. Add `RuntimeBuilder::trust_store`, sealed `Runtime` field and internal
   accessor; change `load_model`/`load_model_observed` signatures.
2. Migrate all seven known call sites plus any `contract_tests` site a
   fresh grep finds once the signature change makes them fail to compile
   (the compiler enumerates every site exhaustively; no site can be missed
   silently).
3. Add the `ModelInstanceError` variants and the `create_model_instance`
   cross-check, plus contract tests.
4. Add `required_weight_shapes` threading and the `stage_weight` check,
   plus tests.
5. Full local verification suite, push, CI, archive, `CHANGELOG.md` update
   -- following this session's established ritual.

No rollback strategy beyond normal git revert is needed: this is a
pre-release library on an unreleased branch, not a deployed service.

## Open Questions

None -- both scope decisions (trust: Runtime-sealed; provenance: cross-check
against the full plan, not just architecture) were made explicitly by the
user before this design was written.
