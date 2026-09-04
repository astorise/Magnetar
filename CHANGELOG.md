# Changelog

## v0.1.0 - Unreleased

### Added

- `magnetar-runtime` local inference runtime baseline.
- `magnetar-cli` boundary harness for local run, chat, model, provider,
  device, and serve workflows.
- Runtime contracts for Component, Capability, Provider, Device, Resource
  Affinity, Resolution Policy, Memory, Tensor, Operator, Kernel Registry,
  Kernel Dispatch, Reference CPU, Model Artifact, Model Loading, Model
  Instance, Tokenizer, Generation, Sampling, Session, KV Cache, Prefix Cache,
  Continuous Batching, Runtime Inference API, and E2E conformance.
- Release packaging, release security, release conformance, and v0.1 cutover
  policy modules with executable validation coverage.

### Changed

- Component and Model artifact trust decisions now fail closed for
  self-asserted publisher/source identity. Digest pinning and explicit local
  development policy remain the accepted non-signature trust mechanisms.
- README implementation status now separates implemented contracts, fixture
  paths, preview behavior, deferred roadmap items, and unsupported v0.1 scope.
- Canonical OpenSpec specs now include release-relevant purpose statements
  instead of archive-generated placeholders.
- First-native Qwen prefill/decode now executes through `ExecutionGraph` plus
  published `PreparedExecutionPlan` bindings, with no hot-path Kernel Registry
  rediscovery once a plan generation is ready.
- First-native compute now resolves its Provider and allocates through the
  Runtime-owned `MemoryManager` exclusively; the production compute path no
  longer constructs a local `ReferenceCpuExecutor` or `MemoryManager` of its
  own.
- Executed model weights are now bound to Runtime resources created by model
  loading and referenced by the active `ModelInstance`, not a fixture-weight
  side channel.
- KV cache data is now a Runtime-owned resource with session/model
  instance/layer/affinity/accounting metadata and transactional prepare,
  commit, and abort semantics tied to sampling and token-commit success.
- Decode now passes the correct absolute position for the token being
  decoded, covered by a multi-step generation-loop oracle.
- Per-step causal observations (Model Instance readiness re-checked every
  step, graph validation) replace evidence booleans that were previously
  fixed at the call site; the Runtime Inference API observation buffer is
  now capacity-bounded rather than unbounded.
- `magnetar chat` now executes every turn of a chat session through one
  persistent Runtime `Runtime` and `InferenceSessionId`; cancellation and
  close act on that same session rather than a session no turn used.
- Production coverage measurement now excludes `first_native_runtime`'s unit
  tests (moved to a sibling `tests.rs` file, matching every other module's
  convention) instead of counting embedded `#[cfg(test)]` source as Runtime
  implementation source.

### Fixed

- Prevented forged Component publisher/source metadata from granting trust.
- Prevented forged Model provenance publisher metadata from granting trust.

### Known Limitations

- The E2E success path still uses a tiny synthetic fixture architecture and
  fixture tokenizer for its numerics, not a published Qwen checkpoint or
  production tokenizer artifact.
- Production model hub downloads, production server API, GPU Providers,
  production CLI UX, and agent/tool Runtime execution are outside v0.1 scope.
- Weight content-digest verification covers `F32` tensors only -- the one
  dtype `magnetar-runtime`'s `host_tensors_from_artifact_bytes` can
  materialize into a `HostTensor` today. `formats/gguf`'s quantized dtypes
  (`Q4_K`, `Q5_K`, `Q8_0`) and `formats/safetensors`' non-`F32` dtypes
  correctly have no digest (they cannot be materialized this way at all
  yet, so no digest of theirs could be checked regardless); extending
  coverage to those needs dequantization/re-encoding support in
  `magnetar-runtime` first, not just a digest. A non-`F32`-declared
  tensor is not therefore unchecked, though: weight materialization
  independently rejects any tensor whose declared `storage_dtype` is not
  `F32`, regardless of digest presence
  (`seal-runtime-model-trust-and-provenance-authority`) -- this
  limitation is about digest *coverage*, not about whether such tensors
  can be silently forged.
- Architecture Freeze #1 is **accepted** at commit `045a536` (2026-09-04,
  CI run https://github.com/astorise/Magnetar/actions/runs/33879062757,
  zero non-`success` jobs confirmed via `gh run view --json
  status,conclusion` and the jobs list directly). The history below is
  kept in full because each round found something real; the commit and
  CI run cited in this first sentence are what to trust as current. (This
  was briefly downgraded to CANDIDATE five times -- after a sixth audit
  round found P0-A open, after a seventh found a narrower P0 in that same
  fix's own `commit` path, after an eighth found both a `ModelInstance`
  semantic-mutability gap and confirmed byte-content provenance was still
  open, after a ninth found `ModelTrustDecision` forgeable, a further
  `ModelInstanceDefinition` clone-identity gap, and that the real format
  parsers still produced no per-tensor digests at all, and after a tenth
  found `ModelTrustStore` itself freely constructible, `create_model_
  instance` never cross-checking caller-supplied architecture/affinity
  against the loading phase's own resolved plan, and weight materialization
  never checking declared shape/dtype independent of digest presence --
  see this section's history for all five fixes.)
  (An earlier point in this history was itself briefly declared
  "accepted" at commit `e7dc45d` -- that run's first pass had one failing
  job, `wasmtime component engine`, on a pre-existing test
  (`pushed_component_package_temp_materialization_is_removed_with_manager`)
  unrelated to this Change (a parallel-test race scanning the shared
  system temp directory for a filename prefix multiple tests use;
  confirmed by local reproduction under the exact CI feature flags,
  passing in isolation, and passing clean on this branch's prior 3 CI
  runs); a rerun of just that job came back green, so this is noted as
  a known pre-existing flake, not fixed as part of this Change). (The P0
  fix itself landed as `f71b346`; that push's own
  CI run failed on `clippy` for an unrelated reason -- a helper this
  same fix added, `check_repeated_load_unload_does_not_accumulate_
  weight_storage`, was missing the `#[cfg(test)]` attribute its sibling
  helpers all carry, so a fresh non-incremental build flagged it as
  dead code; my own local clippy pass had missed this because
  incremental compilation reused a stale artifact. Fixed by `ff06d98`,
  re-verified with `CARGO_INCREMENTAL=0` locally before repushing, and
  confirmed green by the CI run cited above -- this note exists so the
  accepted commit and the fix commit are not silently conflated.) An
  external audit of commit `0197be1` (PR #36) had
  correctly found two P0 gaps in the weight-materialization lifecycle
  that an earlier acceptance declared at that commit missed: (1)
  `ModelInstance` was marked `Ready` immediately on creation, before
  mandatory weights were ever bound, and `acquire_usage`'s own readiness
  check did not inspect weight bindings -- only a deeper, incidental
  graph-dispatch check happened to catch a missing weight, which is not
  the same as the instance's own reported readiness being trustworthy;
  (2) weight materialization was not transactional:
  `Provider.write_tensor` was called before `MemoryManager` admission
  (inverting the intended authority order), a residency registration
  error was silently discarded (`let _ = record_tensor_residency(...)`),
  a mid-loop failure left already-written weights unrolled-back, and
  `unload_model_instance` released `MemoryAllocationId`s but never
  called `Provider.release_tensor` for weight `TensorResourceId`s,
  leaking Provider-owned storage across repeated load/unload cycles.
  Both are now fixed by `openspec/changes/transactional-weight-
  materialization`: creation no longer auto-readies (the caller must
  explicitly reach `Ready` through the new `ModelInstances::mark_ready`,
  used only once every weight is bound); a new
  `WeightMaterializationTransaction` mirrors the KV cache path's
  already-correct admission-first, error-propagating,
  abort-releases-everything pattern; and `unload_model_instance` now
  resolves each released weight's owning Provider generically (via
  `TensorResidency`/`ResourceAffinity`, no hardcoded Provider name in
  the generic Runtime layer) and releases its storage. A follow-up
  revalidation audit of that fix (commit `633e942`) found one further
  gap: `MemoryManager::release(allocation)` only changes the
  `MemoryAllocation`'s own state and never removed the resource's
  `TensorResidency` record, so `tensor_residency()` kept reporting a
  weight as resident indefinitely after both its Provider storage and
  allocation were released -- a metadata leak growing on every failed
  materialization attempt and every load/unload cycle. Fixed by
  `openspec/changes/invalidate-tensor-residency-on-release`: a new
  `MemoryManager::remove_tensor_residency` is now called from both
  rollback and unload, in the order the audit itself recommended
  (release the Provider tensor, then remove the residency record, then
  release the Memory Manager allocation). A third audit round, this
  time a full-scope re-audit rather than a narrow revalidation
  (commit `a4d411b`), found readiness itself was still partly
  caller-declared: `warm_model_instance`'s public
  `ModelInstanceReadinessChecks` defaults `weights_materialized: true`,
  and the Runtime trusted that outright, so a caller using default
  checks against an instance whose weights were never materialized
  could warm it straight to `Ready`; `WarmupPolicy::Disabled`
  compounded this by publishing `readiness: Ready` without the
  lifecycle transition warmup's other policies perform, producing an
  internally inconsistent `lifecycle: Loading, readiness: Ready` state
  that `acquire_usage`/`generation_reference` (checking only readiness,
  never lifecycle) would still accept. **This means the "accepted"
  declaration above was live and technically inaccurate for the window
  between the previous audit round and this fix landing** -- noted here
  rather than silently corrected, per this document's own standard of
  not letting a known-false claim stand uncorrected. Scope was
  clarified directly with the auditor before fixing (their own initial
  reply mixed "necessary for this PR" with "desirable future
  architecture"): fixed by `openspec/changes/runtime-owned-model-
  instance-readiness-authority` without a warmup-API redesign --
  `warm_model_instance` now derives `weights_materialized`,
  `provider_ready`, and `device_ready` from actual Runtime state
  (resource bindings, the Provider registry, the Device list), ANDed
  with the caller's claim; `ModelInstance::validate_readiness` no
  longer lets a computed `Ready` readiness publish while the lifecycle
  hasn't itself reached a state that supports it; and
  `acquire_usage`/`generation_reference` now require both
  `lifecycle.supports_inference_use()` and
  `readiness.accepts_generation()` as a structural safety net.
  `kernel_preparation_ready`/`autotuning_ready` were deliberately left
  caller-supplied (no generic Runtime-side derivation exists for these
  today). The direct `ModelInstanceManager::mark_ready` bypass was
  *not* closed at this point -- **this was a misreading of the
  auditor's own guidance, corrected in the very next audit round
  below, not something the auditor actually authorized leaving open.**
  Verified: full workspace test suite (1,397 tests across the lib, CLI,
  and `contract_tests` integration binaries) passing, `cargo doc
  --locked --workspace --no-deps` with `RUSTDOCFLAGS="-D warnings"`
  clean (a private intra-doc link this Change introduced broke CI's
  `docs` job on the first push; none of this session's other local
  checks run `cargo doc`, so it was only caught by CI -- fixed and
  reverified locally before repushing), coverage ratchet at 78.91%
  (above the 78.89% baseline), `openspec validate --all --strict`
  77/77, live `magnetar run qwen-test "Hello"` unaffected (the
  production path never calls `warm_model_instance`), and the CI run
  cited above. A fourth audit round, a full re-audit of this exact
  fix, found the mark_ready bypass -- and two more real gaps -- still
  open: (P0-A) `ModelInstance.lifecycle`/`.readiness` were still fully
  public mutable fields, so even sealing `mark_ready`/`transition_to`
  would not have stopped `instance.lifecycle = Ready` as a raw field
  assignment; (P0-B) `weights_materialized` only checked
  `resource_bindings.weights` was non-empty, not that entries were
  backed by a real `TensorResidency` record -- the round-1 test suite
  itself demonstrated this, since its own happy-path test inserted an
  unregistered `TensorResourceId` and still passed; (P0-C)
  `provider_ready` only checked a Provider was registered and exposed
  `execution_api()`, not that its own status model reported it
  actually accepting new work. Fixed by `openspec/changes/seal-model-
  instance-readiness-authority`: `mark_ready`/`transition_to`/`warmup`
  and the matching `ModelInstanceManager` wrappers are now `pub(crate)`
  (verified `magnetar-cli` calls none of them directly before sealing);
  `lifecycle`/`readiness` are now `pub(crate)` fields with public
  read-only `lifecycle()`/`readiness()` accessors; `weights_materialized`
  now requires a matching `TensorResidency` per bound weight;
  `provider_ready` now requires
  `Provider::status_snapshot().accepts_new_work_by_default()`. The
  crate's own integration test suite (an external consumer of this
  crate's public API, same as any embedder) was migrated onto
  `warm_model_instance` with real, residency-backed evidence --
  deliberately the same contract a real embedder must use, not a
  parallel test-only escape hatch (a Cargo feature flag for this was
  considered and rejected: it would be included by `--all-features`,
  which this project's own CI uses, reopening the exact gap for any
  downstream consumer who also builds with `--all-features`). Verified:
  full workspace test suite (1,399 tests) passing, `cargo doc` clean,
  wasm32 check clean, coverage ratchet at 78.91% (above baseline),
  `openspec validate --all --strict` 77/77, live `qwen-test` unaffected.
  A fifth audit round -- confirming every round-2 gap was genuinely
  closed before looking for new ones -- found two further real gaps,
  both already contradicting pre-existing canonical spec text
  (spec-correct, code non-compliant): (P0-1) `resume_model_instance`
  still reached `Ready` via `ModelInstance::resume()`'s internal
  `Suspended -> Loading -> Ready` transitions with no call to the
  Runtime-derived readiness check anywhere in between -- state that made
  an instance eligible for suspension could have changed while
  suspended, and resume would jump straight back to `Ready` on stale
  assumptions; (P0-2) `weights_materialized` checked that every *bound*
  weight had a real residency, but never that the bound set was
  *complete* against the loaded manifest's declared tensor inventory
  (`model-loading`'s pre-existing "Qwen Loading Validates Tensor
  Inventory" and "Partial Loading Policy" requirements), nor that a
  residency's claimed Provider had actually received a `write_tensor`
  call for it -- a caller could record a fully legitimate-looking
  `TensorResidency` for a resource no Provider ever wrote. Fixed by
  `openspec/changes/harden-model-instance-readiness-proof`:
  `ModelInstance::resume()` now only reaches `Loading`;
  `resume_model_instance` completes the transition through
  `warm_model_instance`, reusing the same Runtime-derived-evidence path
  every other route to `Ready` uses. `LoadedModelContext` gains
  `required_weight_names`, populated by `ModelLoadingCoordinator::load()`
  from the manifest (no signature change needed on `create_model_instance`
  or `from_loaded_context`); `weights_materialized` derivation now
  requires every declared name to be bound (when known) and resolves
  each bound weight's residency to its recorded Provider, calling
  `read_tensor` to confirm real backing storage rather than trusting the
  residency record alone -- reading actual Provider storage, not new
  Runtime-owned bookkeeping, because any new state gated behind
  `pub(crate)` would reintroduce the "external test crate can't
  legitimately construct evidence" problem round 2 already solved, and
  any publicly-settable flag would just move the forgery surface.
  Verified: full workspace test suite (1,402 tests) passing, `cargo doc`
  clean, wasm32 check clean, coverage ratchet at 78.93% (above
  baseline), `openspec validate --all --strict` 77/77, live `qwen-test`
  unaffected (its production path never calls `warm_model_instance`/
  `resume_model_instance`), and the CI run cited in this note's opening
  sentence. A sixth audit round, again a full re-audit rather than a
  narrow revalidation (HEAD `9939232`), confirmed every round-5 closure
  held and found two further real gaps: (P0-B) the `openspec archive
  model-loading-materializes-weight-resources -y` merge in `9939232`
  itself had *overwritten*, not merged, three normative paragraphs and
  five anti-forgery scenarios already accepted into the canonical
  `openspec/specs/model-instance/spec.md` "Model Instance Readiness"
  requirement with the older, weaker pre-round-3 text the archived
  Change's own delta file carried -- verified by diffing the spec across
  the commit directly (`db9c947` vs `9939232`). Fixed immediately (a
  spec-text restore, not a design question): the deleted paragraphs and
  scenarios are back, merged with -- not replacing -- the archived
  Change's own new "weight materialization state" addition;
  `openspec/specs/model-loading/spec.md` was unaffected by the same
  archive (purely additive there). (P0-A) `LoadedModelContext` and its
  nested `ModelLoadingResidencyPlan` remain fully publicly constructible
  (every field `pub`, no crate-internal constructor) and
  `Runtime::create_model_instance()` trusts a caller-supplied
  `&LoadedModelContext` with no link back to a Runtime-issued record of
  an actual `ModelLoadingCoordinator::load()` run; separately, round 5's
  `weights_materialized` fix proves only that *some* bytes exist in
  Provider storage under each required `TensorResourceId`, not that
  those bytes are this specific instance's own Runtime-authorized
  materialization -- a caller retaining ordinary public access to
  `Provider::execution_api().write_tensor()`,
  `MemoryManager::record_tensor_residency()`, and
  `resource_bindings.weights` can still assemble a passing state by
  hand, confirmed concretely by this crate's own `contract_tests`
  helper (`bind_fake_weight`) doing exactly that. This gap is
  architecturally larger than any prior round's (comparable to round
  2's field-sealing in kind, larger in the new Runtime-owned evidence
  machinery it needs) and was not fixed inline: scoped into
  `openspec/changes/bind-model-loading-evidence-to-validated-artifact`
  (proposal, design, spec deltas, and a task list are complete; code is
  not yet implemented) per explicit direction, rather than rushed.
  **Architecture Freeze #1 therefore remains CANDIDATE, not accepted,
  until that Change's implementation lands and passes full verification
  on its own final HEAD.** This round's own fix (P0-B only) verified:
  `openspec validate --all --strict` 77/77 (76 canonical items plus the
  new active Change) passing at commit `f57ecde`. P0-A was fixed by
  `openspec/changes/bind-model-loading-evidence-to-validated-artifact`:
  `LoadedModelContext`/`ModelLoadingResidencyPlan` fields became
  `pub(crate)` (Runtime-issued, no longer externally constructible --
  confirmed by grep that no external caller depended on direct
  construction or field access, a materially smaller blast radius than
  round 2's field-sealing);
  `ModelInstanceDefinition.resource_bindings` and all four
  `ModelInstanceResourceBindings` fields became `pub(crate)` (sealing both
  direct-field mutation and whole-field replacement -- cloning one
  instance's bindings onto another, which sealing only the inner fields
  would not have closed); the previously-private
  `materialize_model_instance_weights` became the one public entrypoint
  for turning weight bytes into bound resources, and
  `WeightMaterializationTransaction::commit` now mints a Runtime-owned
  `MaterializationEvidence` (artifact id + the full current committed
  resource-id set) per instance, keyed by `ModelInstanceId` so evidence
  cannot transfer across instances regardless of whether they share an
  artifact. `derive_effective_readiness_checks`'s `weights_materialized`
  now requires this evidence to match alongside `TensorResidency`
  presence, replacing the `read_tensor` Provider-storage probe entirely --
  closing the companion P1 (a device-only Provider without host readback
  can now still prove materialization) as a direct consequence, proven by
  a dedicated test using a Provider double that never overrides
  `write_tensor`/`read_tensor` from their documented no-op/`None`
  defaults. `contract_tests/model_instance.rs`'s `bind_fake_weight` --
  which the audit found was already performing the exact hand-assembled
  forgery it described (real `write_tensor` and real `TensorResidency`,
  but binding and marking Ready by direct field access rather than
  through the authorized transaction) -- was migrated onto the public
  entrypoint; this changed its behavior (the transaction bundles bind,
  evidence, and `mark_ready` in one step, matching the pre-existing
  production contract), which surfaced and required fixing an
  `Ready -> Ready` invalid-lifecycle-transition trap in two dependent
  tests that had previously relied on a separate, now-redundant
  `warm_model_instance` call. Four new tests cover hand-assembled bindings
  without evidence, evidence copied from another instance, evidence after
  an artifact reassignment, and the device-only-Provider happy path.
  Verified: full workspace test suite (1,160 lib tests + 182
  `contract_tests`) passing, `cargo doc` clean (one private intra-doc link
  fixed, the same class of issue round 3 hit), wasm32 check clean,
  coverage ratchet at 78.97% (above the 78.89% baseline), `openspec
  validate --all --strict` 76/76 after archiving (diffed the canonical
  specs across the archive commit -- the same check that caught `9939232`'s
  regression earlier this round -- and confirmed the merge was purely
  additive plus the one intended paragraph rewrite, nothing silently
  dropped), live `magnetar run qwen-test "Hello"` unaffected (real
  generation completed, matching output shape to every prior round), and
  the CI run cited in this note's opening sentence. **Deliberately not
  closed by this fix, and out of this Change's stated scope from the
  start:** byte-content provenance -- proving materialized bytes are
  bit-identical to the specific validated Model Artifact's declared tensor
  content, beyond the one existing fixture-specific digest check in
  `bind_qwen_fixture_weights` -- would need a manifest-level per-tensor or
  aggregate digest threaded from artifact parsing through
  `LoadedModelContext` (the same shape of plumbing `required_weight_names`
  already established) and verified inside the transaction; this touches
  the `model-artifact`/`model-loading` manifest schema, a different
  capability boundary than `model-instance` readiness, and is proposed as
  its own future Change rather than folded in here.
  **Correction, made when an eighth audit round questioned it directly:**
  the sentence originally here claimed "the round-6 audit's own P0
  classification was specifically about caller-constructible evidence,
  not cryptographic content verification" -- re-reading round-6's actual
  text (its sections 12-16) shows this was not accurate. Round 6 explicitly
  treated "Provider weight bytes are not provenance-bound to the validated
  Model Artifact" as the second half of the *same* P0-A, recommending both
  halves "be treated as one architectural P0," not as a separate,
  lower-priority concern. Scoping byte-content provenance out of that
  Change's actual implementation was this session's own decision (recorded
  in the Change's own design.md Non-Goals, for real reasons: smaller blast
  radius, avoiding a second large schema change bundled into an already-
  large fix), not something round-6 itself endorsed as sufficient on its
  own. A seventh audit round reviewed that scoping decision and accepted it
  (classifying the residual as P1); an eighth round revisited the same
  question and did not accept it, citing round-6's original text -- a
  legitimate disagreement between two revalidation passes about how much
  deferral round-6's finding could bear, not a new fact either time. Given
  this, and given cryptographic byte-content provenance was never actually
  implemented, it is more honest to say: **byte-content provenance was
  part of round-6's original P0 and has not yet been closed** -- see the
  ninth-round paragraph below for the OpenSpec Change now implementing it
  (`bind-materialized-weight-content-to-model-artifact-digests`). A seventh audit round (HEAD
  `ef0e9e2`), revalidating that exact fix, confirmed every round-6 closure
  held (`LoadedModelContext`/bindings sealing, instance-scoped evidence,
  artifact-id matching, the removed `read_tensor` dependency, the restored
  canonical spec text) and found one further real gap in the fix's own
  new code: `WeightMaterializationTransaction::commit`
  (`first_native_runtime.rs`) minted `MaterializationEvidence` and called
  `ModelInstanceManager::mark_ready` unconditionally right after
  publishing whatever bindings this attempt staged -- so
  `materialize_model_instance_weights(..., &BTreeMap::new())` against an
  instance with a non-empty mandatory weight inventory could still reach
  `Ready`, and a strict subset of a multi-tensor inventory produced
  evidence that was exactly correct for what it staged while still being
  incomplete. Verified directly against the code and the canonical spec
  before accepting this: `model-loading`'s pre-existing "Model Loading
  Does Not Bypass Instance Readiness" ("Successful materialization alone
  SHALL not imply Model Instance readiness") and "Partial Loading Policy"
  requirements already prohibited exactly this -- the code was simply
  non-compliant with spec text that already existed, so (matching the
  audit's own classification) this was fixed directly, with no new
  OpenSpec Change. `commit` now gates the `mark_ready` call behind the
  same Runtime-derived readiness gate `warm_model_instance`/
  `resume_model_instance` already use (`derive_effective_readiness_checks`,
  promoted from private to `pub(crate)` for this) via
  `ModelInstance::validate_readiness`, rather than a parallel or weaker
  check; progressive/incremental materialization across multiple calls
  still works unchanged, since evidence is recomputed against the full
  current binding set on every commit. Four new tests prove this at the
  exact public entrypoint the bug was in -- empty map, partial inventory
  (plus completing it in a follow-up call), and a Provider that rejects
  new work -- since `warm_model_instance`'s own tests, though already
  correct, could not by themselves prove this separate bypass was closed.
  Verified: full workspace test suite (1,163 lib tests + 182
  `contract_tests`) passing, `cargo clippy`/`cargo fmt --check` clean,
  `cargo doc` clean, wasm32 check clean, coverage ratchet at 78.97%
  (above baseline), `openspec validate --all --strict` 76/76 (unchanged --
  this fix touched no spec text), live `magnetar run qwen-test "Hello"`
  unaffected, and the CI run cited in this note's opening sentence
  (commit `44ae71e`). An eighth audit round (HEAD `44edcad`), a full
  re-audit, confirmed every round-7 closure held and found two further
  real gaps -- one genuinely new (P0-1), and one that turned out to be a
  correction of this document's own prior wording rather than a new code
  defect (P0-2, addressed above where that correction lives). (P0-1)
  `ModelInstance.definition` was still a fully public mutable field, so
  every semantic property of an already-`Ready` instance -- `artifact`,
  `architecture`, `placement`, `policy`, `tokenizer`,
  `kernel_selection_policy`, ... -- could be reassigned directly by any
  external caller, with no immediate invalidation:
  `acquire_usage`/`generation_reference` only check lifecycle/readiness,
  never whether the definition still matches what was evidenced. Worse,
  `ModelInstanceDefinition`'s `#[derive(Clone)]` copies every field --
  including the already-sealed `resource_bindings` from round 6 -- since
  `Clone`'s generated code runs with full in-crate access regardless of
  the caller's own field visibility; a caller could clone an
  already-`Ready` instance's definition (carrying real weight bindings
  pointing at live Provider resources) into a *new* instance via
  `ModelInstanceManager::create` (directly, or through `reload`, which
  calls `create` internally), then call
  `materialize_model_instance_weights` with an empty weights map --
  `commit` minted fresh evidence over the instance's full *current*
  binding set (inherited from the clone, not staged by this attempt),
  aliasing a real Provider resource across two distinct
  `ModelInstanceId`s. This is a resource-ownership break, not only a
  metadata inconsistency: unloading the new instance would release a
  resource the original instance still owned. Fixed by sealing
  `ModelInstance.definition` to `pub(crate)` with a read-only
  `definition()` accessor (two narrow post-creation methods,
  `set_provider_resource` and `track_memory_allocation`, cover the only
  legitimate external mutation needs this crate's own test suite had,
  neither able to affect readiness, generation, or Provider/Device
  resolution), and by having `ModelInstanceManager::create` reset
  `resource_bindings` unconditionally regardless of what the supplied
  definition carried -- the shared chokepoint both `create` and `reload`
  go through, so bindings can only ever be populated again by that
  specific instance's own future `commit` calls. Direct fix against
  already-correct canonical spec text ("Runtime owns Model Instance",
  silent semantic mutation forbidden); no new OpenSpec Change needed.
  Two new tests prove both the `create()` and `reload()` paths are closed
  at the exact public entrypoints the audit described. Verified: full
  workspace test suite (1,163 lib tests + 184 `contract_tests`) passing,
  `cargo clippy`/`cargo fmt --check` clean, `cargo doc` clean, wasm32
  check clean, coverage ratchet at 78.97% (above baseline), `openspec
  validate --all --strict` 76/76 (unchanged), live `magnetar run
  qwen-test "Hello"` unaffected, and CI run
  https://github.com/astorise/Magnetar/actions/runs/33838701914 (commit
  `539e723`) confirmed green via `gh run view --json status,conclusion`
  and the jobs list directly. The same eighth audit round also revisited
  byte-content provenance (P0-2) and, reading round-6's original text
  directly, disputed this document's own prior claim that round-6's P0
  was "specifically about caller-constructible evidence, not
  cryptographic content verification" -- that characterization is
  corrected earlier in this history rather than silently rewritten.
  Closed by `openspec/changes/bind-materialized-weight-content-to-model-
  artifact-digests`: `ModelTensorMetadata` gains an optional per-tensor
  content digest field (mirroring the existing whole-artifact/part/shard
  digest fields), parsed from manifest YAML the same way the
  artifact-level digest already is; `LoadedModelContext`/
  `ModelInstanceDefinition` gain `required_weight_digests`, threaded
  exactly the way `required_weight_names` already is (populated once at
  `ModelLoadingCoordinator::load()` time, empty means unknown, no
  signature changes); `WeightMaterializationTransaction::stage_weight`
  verifies each staged tensor's bytes against its declared digest (via a
  new `HostTensor::content_bytes()` canonical byte representation)
  before Memory Manager admission, returning the new
  `InferenceApiError::WeightContentDigestMismatch` on a mismatch through
  the existing rollback path. The check is generic (any manifest that
  declares digests), not fixture-specific -- `formats/gguf`/`formats/
  safetensors` (confirmed real, but not yet digest-populating, format
  parsers, unlike `model_format_roadmap.rs`'s still-unimplemented
  roadmap contracts) need no rework to benefit from it once they compute
  digests of their own. The E2E fixture's checked-in `.safetensors` file
  is the one real artifact source populated with real digests today --
  `e2e_fixture_manifest` now builds its tensor YAML from
  `e2e_fixture_weight_inventory_with_digests`, computing each digest
  from the real file's actually-parsed bytes. Two new tests prove the
  exact entrypoint the gap was in (tampered content rejected with the
  specific new error; real content still materializes normally); a
  pre-existing test (`check_weight_byte_change_alters_generated_logits`,
  proving mutated weight bytes are numerically consumed) needed a
  digest-free clone of the fixture manifest to keep working, since its
  own deliberately-mutated bytes would otherwise now be correctly
  rejected by the new check -- isolating that test's orthogonal concern
  from this Change's. Shipping this surfaced a real, separate gap in my
  own initial change-scope check: `ModelTensorMetadata` is also
  constructed directly by the `formats/gguf` and `formats/safetensors`
  git submodules (`astorise/Magnetar-format-GGUF`,
  `astorise/Magnetar-format-safetensors`), which a grep scoped to
  `magnetar-runtime/` alone did not reach; CI's "format integration" and
  "submodule integration" jobs caught the resulting missing-field
  compile error on the first push (commit `61c6061`), fixed in both
  submodules (each defaulting the new field to `None`, matching its
  "absent means unknown" semantics) and the main repo's submodule
  pointers bumped (commit `08f9178`). Verified: full workspace test
  suite (1,165 lib tests + 184 `contract_tests`) passing, `cargo
  clippy`/`cargo fmt --check` clean, `cargo doc` clean, wasm32 check
  clean, coverage ratchet at 79.00% (above baseline), `openspec validate
  --all --strict` 76/76 (unchanged by the submodule fix), live `magnetar
  run qwen-test "Hello"` unaffected (confirming the real fixture's real
  digests match its real materialized bytes in production, not only in
  tests), each submodule's own test suite passing independently (16/16
  each), and the CI run cited in this note's opening sentence (commit
  `08f9178`). A ninth audit round (HEAD `7908e29`), a full re-audit,
  confirmed every round-8 closure held and found three further real gaps,
  all matching the same "field sealing closes construction, but not
  Clone-and-carry, and only covers what was actually reset" root cause
  this session has now seen several times. (P0-A) `ModelTrustDecision`'s
  fields and constructor were still fully `pub`. Model Loading's public
  `load_model`/`load_model_observed` accept a caller-supplied
  `&ModelTrustDecision` directly, and `validate_preconditions` only
  switches on `trust.status` -- it never re-derives trust from a
  `ModelTrustStore`. So an external caller could construct
  `ModelTrustDecision::new(Trusted, "...")` directly and pass it straight
  through as if a real trust store had evaluated it, entirely bypassing
  the one fail-closed mechanism that actually exists
  (`ModelTrustStore::evaluate`, already correctly used by `magnetar-cli`'s
  own production code at `commands.rs:714`, but never enforced at the
  public API boundary itself). Fixed by sealing the constructor and
  fields to `pub(crate)` with `status()`/`reason()` read accessors; the
  only way to obtain a `Trusted` decision is now
  `ModelTrustStore::evaluate`. Confirmed via grep before making the
  change: only 4 external (`contract_tests`) call sites existed, all
  constructing fixtures, none reading the fields directly except in one
  file this grep initially missed (`contract_tests/model.rs`, caught by
  the build) -- migrated onto `ModelTrustStore::default().trust_digest(...)
  /.reject_digest(...).evaluate(...)`. (P0-B) `ModelInstanceDefinition.
  artifact`/`.architecture` were still fully `pub`, so a caller could
  clone an existing instance's `definition()` (round 8's read-only
  accessor), reassign `.artifact` to a different `ModelArtifactId`, and
  pass the mutated clone to the still-public `ModelInstanceManager::
  create` -- publishing a new instance declaring an artifact identity
  Model Loading never actually validated, while inheriting the original's
  `required_weight_names`/`required_weight_digests` (`pub(crate)`, but
  `Clone` copies them regardless of field visibility, the exact reasoning
  round 8 already established for `resource_bindings`). Round 8's fix
  (resetting `resource_bindings` in `create()`) correctly stopped the
  clone from inheriting *resources* but not from inheriting or rewriting
  *identity*. Fixed by sealing both fields to `pub(crate)` -- confirmed
  via grep, zero external blast radius. Both P0-A and P0-B are direct
  fixes against already-correct canonical spec text, per the audit's own
  governance framing; no new OpenSpec Change needed for either. (P0-C)
  The real `formats/gguf`/`formats/safetensors` submodules -- confirmed
  real parsers, not `model_format_roadmap.rs`'s still-unimplemented
  roadmap contracts -- always set `digest: None` on every tensor they
  produced, so `bind-materialized-weight-content-to-model-artifact-
  digests`'s content verification, implemented in round 8, never actually
  constrained materialized weight bytes for a real Safetensors/GGUF
  artifact, only the in-repo E2E fixture. Fixed in both submodules: for
  `F32` tensors (safetensors) and unquantized `F32` tensors
  (`ggml_type == 0`, GGUF) -- the one dtype `magnetar-runtime`'s
  `host_tensors_from_artifact_bytes` materializes into a `HostTensor`
  today -- each parser now computes a real `ModelDigest::sha256` over the
  tensor's actual raw file bytes; both formats' `F32` on-disk layout
  (raw, native-width, little-endian IEEE754) is already byte-identical to
  `HostTensor::content_bytes()`'s canonical representation, so this
  cannot mismatch what the transaction verifies later. Non-`F32`/quantized
  dtypes correctly stay `digest: None` -- they cannot be materialized
  into a `HostTensor` without a re-encoding or dequantization step neither
  crate performs, so no digest of theirs could ever be checked anyway (a
  genuine, not merely deferred, limitation -- see this section's own
  entry above). Two new tests per submodule prove a real digest verifies
  against its own bytes and rejects tampering, and that non-materializable
  dtypes correctly have none. Verified: both submodules' own test suites
  (18/18 each, +2 from this fix), full workspace test suite (1,165 lib
  tests + 184 `contract_tests`, unchanged counts -- P0-A/P0-B needed no
  new runtime tests, matching the compile-time-guarantee precedent round
  6's `LoadedModelContext` sealing already established) passing, `cargo
  clippy`/`cargo fmt --check` clean, `cargo doc` clean, wasm32 check
  clean, coverage ratchet at 79.00% (above baseline), `openspec validate
  --all --strict` 76/76 (unchanged -- none of the three fixes touched
  spec text), live `magnetar run qwen-test "Hello"` unaffected, and the
  CI run cited in this note's opening sentence (commit `e2379eb`, main
  repo; submodule commits `5fc5ac7`/`da890d9`).
  A tenth audit round (commit `7269290`) found that each of the ninth
  round's sealed-forgery fixes had a sibling gap one level up, reachable
  through code that stayed fully `pub`: (P0-A) `ModelTrustStore` itself
  had every field `pub` and a fully public builder/evaluation API -- any
  caller could construct their own store, self-declare a digest trusted,
  and obtain a real (not forged) `Trusted` decision for whatever artifact
  they chose before calling `load_model`; sealing `ModelTrustDecision::
  new` (round 9) closed fabricating a decision directly, not this.
  Investigating the fix surfaced that `Runtime` held no
  `ModelLoadingCoordinator` anywhere in the codebase -- every real call
  site (CLI, the qwen-test live E2E fixture, both test modules) built one
  standalone and only wired the result into a separate `Runtime`
  afterward -- so sealing trust for real required coupling `load_model`
  to a `Runtime`-owned, once-configured policy, not narrowing
  `ModelTrustStore`'s visibility. Fixed: `RuntimeBuilder::trust_store`
  (set once, no post-build reconfiguration, default trusts nothing);
  `load_model`/`load_model_observed` take `&mut Runtime` and evaluate
  trust internally, with no parameter through which a caller can still
  supply a decision. (P0-B) `Runtime::create_model_instance` accepted
  caller-supplied `architecture`/`affinity` never compared against
  `loaded.plan()`'s own resolved values. Fixed: rejects an architecture
  identity disagreeing with the loading phase's resolved architecture,
  and an affinity naming a provider/device disagreeing with one the
  loading phase resolved (an unresolved plan field imposes no
  constraint, consistent with this crate's `None`-is-permissive
  precedent); new `ModelInstanceError::ArchitectureMismatch`/
  `AffinityMismatch`. (P0-C) Weight materialization only ever checked a
  content digest, and digests are `F32`-only (the known limitation noted
  above) -- so a tensor the manifest declared quantized had no digest,
  letting a caller fabricate `F32` content under its name with nothing to
  reject it, silently bypassing the format parser's correct refusal to
  materialize non-`F32` content. Fixed: `stage_weight` now also checks
  the manifest's declared shape and that the declared `storage_dtype` is
  `F32`, independent of digest presence and preceding the digest check;
  new `InferenceApiError::WeightShapeOrDtypeMismatch`. Implementing this
  check surfaced a real, previously-unnoticed inconsistency in
  `contract_tests::model_instance`'s generic fixture (a `bf16`-declared
  tensor whose test helper fabricated fake `F32` content under its name)
  -- fixed by aligning the fixture, not by loosening the check. All three
  tracked as `seal-runtime-model-trust-and-provenance-authority`, archived
  at `2026-09-04-seal-runtime-model-trust-and-provenance-authority`.
  Verified: full workspace test suite (64 + 1,171 lib tests [+6 from this
  fix] + 184 `contract_tests`) passing, `cargo clippy`/`cargo fmt --check`
  clean, `cargo doc` clean, wasm32 check clean, coverage ratchet at
  79.00% (above baseline), `openspec validate --all --strict` 76/76
  before archiving (75/75 canonical after), live `magnetar run qwen-test
  "Hello"` unaffected, and the CI run cited in this section's opening
  sentence (commit `045a536`).
- Release artifacts are not final until generated from the exact release commit
  and tag.

### Security Notes

- Cryptographic artifact signing is not implemented for v0.1 and must not be
  claimed as present; it is deferred to a dedicated future change and this
  release makes no authenticated publisher identity claim.
- Cache presence, model alias, local file location, recognized format,
  publisher metadata, source metadata, and fixture markers do not grant trust by
  themselves.
