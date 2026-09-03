## 1. Spec

- [x] 1.1 `model-loading`: new "Loaded Model Context Is Runtime-Issued" requirement.
- [x] 1.2 `model-instance`: "Model Instance Readiness" strengthened for Runtime-issued materialization evidence; new "Model Instance Weight Resource Bindings Are Runtime-Sealed" requirement. `openspec validate --strict bind-model-loading-evidence-to-validated-artifact` passes.

## 2. Materialization evidence storage

- [x] 2.1 Added `MaterializationEvidence` (artifact id + committed resource-id set, `matches()` comparator) and `ModelInstanceManager`-owned storage (`materialization_evidence: BTreeMap<ModelInstanceId, MaterializationEvidence>`, with `record_materialization_evidence`/`materialization_evidence`/`clear_materialization_evidence`, all `pub(crate)`) -- additive only.
- [x] 2.2 `cargo build -p magnetar-runtime --lib` clean (only expected dead-code warnings, resolved once wired up in group 4/6).

## 3. Seal `LoadedModelContext` (audit P0-A, loading half)

- [x] 3.1 `LoadedModelContext`/`ModelLoadingResidencyPlan` fields become `pub(crate)`.
- [x] 3.2 `cargo build --workspace --all-targets --all-features` clean for this step in isolation, confirming the design's grep was right that no call site outside `model_loading.rs` needed a code change for construction. **Correction discovered one step later (5.4 below): the design's grep (pattern `loaded\.<field>`) missed `contract_tests/model_loading.rs`, which reads (not constructs) `context.state`/`context.plan.*` under a differently-named local variable -- a real external read dependency the grep's variable-name assumption hid.**

## 4. Public materialization entrypoint + evidence minting (audit P0-A, provenance half)

- [x] 4.1 Made the existing private free function `materialize_model_instance_weights` `pub` directly (not a new `Runtime` method wrapper) -- matches this module's existing style for its other public free functions (`register_qwen_component_artifact`, `run_first_native_fixture_generation`) and is re-exported at the crate root via `lib.rs`'s existing `pub use first_native_runtime::*;`, so no new export wiring was needed. Same `(instance, artifact_owner, weights)` signature, unchanged.
- [x] 4.2 `WeightMaterializationTransaction::commit` now mints/replaces the instance's `MaterializationEvidence` after publishing this attempt's staged bindings, using the artifact id from `definition.artifact` and the **full current** `resource_bindings.weights` value set read back after the insert (not just this attempt's own staged subset) -- see 4.3's correction for why.
- [x] 4.3 **Correction from design.md's plan:** only `Runtime::unload_model_instance` clears evidence; `WeightMaterializationTransaction::abort` does **not**. Reasoning found during implementation: `commit()` inserts are additive/upsert (a second materialization attempt can add more weights to an already-partially-materialized instance), so if a *later* attempt fails and aborts, clearing evidence would wrongly invalidate an *earlier* attempt's still-valid, already-committed materialization -- `abort` only ever rolls back its own attempt's newly staged (not-yet-committed) resources, never persisted state. Recomputing evidence from the full current binding set on every successful `commit` (4.2) makes this correct without needing `abort` to touch evidence at all.

## 5. Seal weight resource bindings (audit P0-A, provenance half)

- [x] 5.1 `ModelInstanceDefinition.resource_bindings` (the whole field, per design.md's whole-field-replacement/clone-reuse concern) and all four `ModelInstanceResourceBindings` fields become `pub(crate)`.
- [x] 5.2 `contract_tests/model_instance.rs`'s `bind_fake_weight` migrated onto the step-4 public entrypoint. **Design refinement discovered mid-implementation:** `materialize_model_instance_weights` bundles bind+evidence+`mark_ready` in one call (matching production's `bind_qwen_fixture_weights`), so every existing caller that did `bind_fake_weight` then separately called `warm_model_instance` would now attempt an invalid `Ready -> Ready` lifecycle transition (no such transition exists in `ModelInstanceLifecycleState::allows_transition_to`) and fail. Fixed by removing the now-redundant follow-up `warm_model_instance` call from `reach_ready` (it becomes just `bind_fake_weight`) and from the one other direct call site that hit this (replaced with a direct `lifecycle() == Ready` assertion); the one remaining direct call site (the `adapter_ready: false` failure-path test) needed no change, since `warmup()`'s failure branch sets `lifecycle = Failed` as a raw assignment, not through `transition_to`'s transition-table check, so it is unaffected by the starting lifecycle. Also added a new narrow public `ModelInstanceDefinition::track_memory_allocation(&mut self, allocation: MemoryAllocationId)` for one test's legitimate need to track an out-of-band Memory Manager allocation unrelated to weight materialization (does not touch `weights` or evidence, so it cannot be used to forge `weights_materialized`).
- [x] 5.3 `cargo build --workspace --all-targets --all-features` clean.
- [x] 5.4 Two real external read dependencies surfaced (see 3.2's correction), both in `contract_tests/model_loading.rs`: added public read-only accessors `LoadedModelContext::state()`/`::plan()` and `ModelLoadingResidencyPlan::quantization_handling()`/`::memory_placements()`; updated the two call sites to use them.

## 6. Readiness derivation switch

- [x] 6.1 `derive_effective_readiness_checks`'s `weights_materialized`: replaced the `ProviderExecutionApi::read_tensor` probe with `runtime.model_instances().materialization_evidence(instance)` matched via `MaterializationEvidence::matches(&definition.artifact, &bindings.values().cloned().collect())`.
- [x] 6.2 Confirmed: `derive_effective_readiness_checks` no longer calls `read_tensor` at all. Other call sites remain (actual tensor computation in kernel dispatch/graph execution), unaffected and out of scope -- only the readiness-proof usage is removed.

## 7. Tests

- [x] 7.1 Forged-context rejection: this is a compile-time property, not a runtime test -- `LoadedModelContext`'s sealed fields make external construction impossible, confirmed by `contract_tests` already only ever obtaining one via `ModelLoadingCoordinator::load()`. No same-crate `tests.rs` test added: `pub(crate)` is crate-wide, so a same-crate test constructing one directly would not prove anything about external-caller forgery resistance (matching the established `lifecycle`/`readiness` precedent that same-crate access is intentional, not a gap).
- [x] 7.2 Added `inference_api_warm_model_instance_rejects_hand_assembled_binding_without_evidence`: real `write_tensor` + real `record_tensor_residency` + direct (same-crate) binding, no evidence minted -- rejected.
- [x] 7.3 Added `inference_api_warm_model_instance_rejects_bindings_copied_from_another_instance`: instance A materializes for real; instance B's bindings are set to match A's exactly (same artifact, same resource ids) -- B still rejected (evidence is instance-keyed, not artifact- or binding-keyed), A remains unaffected and still Ready.
- [x] 7.4 Added `inference_api_warm_model_instance_rejects_evidence_after_artifact_reassignment`. **Correction to design.md's own note:** `artifact` does not need a dedicated setter to be reachable -- it is a plain `pub` field on `ModelInstanceDefinition` (never sealed by this Change), so direct reassignment after `Ready` is a real, not just hypothetical, path; proven directly rather than left as defense-in-depth-only. Confirmed this cannot be used to *gain* unauthorized Ready status (evidence lookup stays instance-keyed per 7.3), only to invalidate an instance's own otherwise-valid evidence -- fail-closed, not a new forgery vector, so `artifact` was correctly left unsealed.
- [x] 7.5 Added `inference_api_warm_model_instance_reaches_ready_without_provider_read_tensor_support`, using `TestProviderExecutionApi` registered *as* `REFERENCE_CPU_PROVIDER_NAME` (the one Provider name the transaction is hardcoded to, matching v0.1 baseline scope) -- it does not override `write_tensor`/`read_tensor`, inheriting the trait's real default bodies (no-op write, `None` read), so it is a faithful device-only-Provider stand-in. The test asserts `read_tensor` genuinely returns `None` before relying on that fact, so a future accidental override would be caught rather than silently passing for the wrong reason.
- [x] 7.6 Existing test suite audit, in `tests.rs`: `reach_ready_with_real_weight` migrated onto `materialize_model_instance_weights`, fixing 3 callers. `..._rejects_incomplete_mandatory_weight_inventory` rewritten to use real materialization for the one bound weight (isolating the inventory-completeness check specifically, via `warm_model_instance` re-derivation). `..._rejects_residency_without_provider_write` left structurally unchanged (still compiles and passes under same-crate `pub(crate)` access) with a doc-comment note that it now incidentally also exercises the evidence check, kept anyway since it still tests the residency-presence axis independently. In `contract_tests/model_instance.rs`: `bind_fake_weight`/`reach_ready` migrated onto the public entrypoint (see group 5's task 5.2 note for the lifecycle-transition correction this required).
- [x] 7.7 Live production path (`bind_qwen_fixture_weights` -> `materialize_model_instance_weights` -> `WeightMaterializationTransaction::commit`) unchanged in this Change beyond `commit` now also minting evidence -- no different bindings, no different lifecycle transition, no different public signature. Confirmed by inspection here; the live `qwen-test` run in group 8 is the empirical check.

## 8. Verification

- [x] 8.1 `cargo build --locked --workspace --all-targets --all-features` clean.
- [x] 8.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` clean.
- [x] 8.3 `cargo fmt --all -- --check` clean (one real diff found and applied: `rustfmt` reflowed `MaterializationEvidence::new`/`matches` and the `clear_materialization_evidence` call onto multiple lines).
- [x] 8.4 `cargo test --locked --workspace --all-targets --all-features`: 1160 lib tests + 182 contract_tests passing, 0 failed (lib count up from 1156 baseline by the 4 new tests in group 7).
- [x] 8.5 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean (one private intra-doc link found and fixed: `[`WeightMaterializationTransaction::commit`]` -> plain backticks, the same class of issue round 3 hit).
- [x] 8.6 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean (exit 0; pre-existing unrelated warnings in `first_native_runtime.rs`'s component-engine-unavailable dead code paths, not touched by this Change).
- [x] 8.7 Coverage ratchet: 78.97% (above the 78.89% baseline).
- [x] 8.8 `openspec validate --all --strict`: 77/77 passing (76 canonical + this active Change).
- [x] 8.9 Live `magnetar run qwen-test "Hello"` unaffected: real generation completed, `[generated token ids: 239 239 239 239]`.
- [ ] 8.10 Push, confirm a full green CI run via `gh run view --json status,conclusion` and the jobs list directly (never a piped `gh run watch`).

## 9. Close out

- [ ] 9.1 Diff the canonical spec files this Change touches before and after archiving (matching the exact check this Change's own P0-B discovery used) to confirm the archive-merge did not silently drop any requirement or scenario.
- [ ] 9.2 Archive this Change.
- [ ] 9.3 Update `CHANGELOG.md`'s Architecture Freeze #1 note: this Change closing means P0-A is resolved; note explicitly that byte-content provenance (this Change's Non-Goal) remains open and name the proposed follow-up.
- [ ] 9.4 Consider whether to open the follow-up Change (manifest-level per-tensor/aggregate content digest verification) now or defer -- a decision for the user, not made here.
