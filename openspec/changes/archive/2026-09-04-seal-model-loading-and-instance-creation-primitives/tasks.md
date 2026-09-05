## 1. Spec

- [x] 1.1 `model-loading`: new "Model Loading Is Reachable Only Through the Runtime-Sealed Loading API" requirement.
- [x] 1.2 `model-instance`: "Model Instance References Architecture Implementation" corrected to describe the unresolved-Provider/Device case as a documented limitation, not a caller-authority grant; new "Model Instance Creation Is Reachable Only Through Runtime" requirement.
- [x] 1.3 `inference-api`: "Provider Preferences Are Non-Authoritative" gains a cross-reference to the same documented limitation. `openspec validate --strict seal-model-loading-and-instance-creation-primitives` passes.

## 2. Confirm no non-test caller depends on either primitive

- [x] 2.1 Confirmed zero non-test hits in `magnetar-cli`, `formats/gguf`, `formats/safetensors` for all three primitives before sealing.

## 3. Seal `ModelLoadingCoordinator::load`

- [x] 3.1 `pub fn load` -> `pub(crate) fn load` (`model_loading.rs`).
- [x] 3.2 `cargo build`/`cargo test` enumerated every break: 5 sites in `contract_tests/model_loading.rs`.
- [x] 3.3 Relocated those 5 tests (untrusted rejection, Ready-context shape, memory-budget/quantization/allocation failure mapping) into `magnetar-runtime/src/tests.rs` as `sealed_loading_*`.
- [x] 3.4 What remained in `contract_tests/model_loading.rs` (`resolve_architecture`, `ModelLoadingState::can_transition_to`, the standalone dtype/cache/reload helpers) stays there unchanged -- confirmed genuinely `load`-independent.

## 4. Seal `ModelInstanceDefinition::from_loaded_context` and `ModelInstanceManager::create`

- [x] 4.1 Both `pub fn` -> `pub(crate) fn` (`model_instance.rs`). Also sealed `ModelInstanceManager::create_checked` (`#[cfg(test)]`-gated, not found until implementation): a second, equally public path to the same `create` bypass that the original design and the audit itself had not named -- confirmed zero non-test callers anywhere in the crate before sealing.
- [x] 4.2 `cargo build`/`cargo test` enumerated 28 break sites across `contract_tests/model_instance.rs` -- more than the design's initial 2-test estimate for D2's "genuinely tests the primitive's own contract" category.
- [x] 4.3 Relocated 6 tests into `magnetar-runtime/src/tests.rs` as `sealed_creation_*` (not 2, as first estimated): `cloned_definition_does_not_inherit_weight_authority`, `reload_replacement_does_not_inherit_original_weight_authority` (as planned), plus `creation_and_readiness_checks_gate_ready_state` (tests `create_checked`'s own checks-validation, discovered needing sealing during 4.1), `adapter_activation_records_mutation_and_invalidates_dependent_caches`, `unload_releases_memory_provider_resources_adapters_and_cache_dependencies`, and `browser_policy_rejects_native_or_oversized_instance_features` (all three pre-populate definition fields -- `usage.kv_cache_dependencies`, `associated_sessions`, `policy.browser_linear_memory_limit_bytes` -- before creation, which `Runtime::create_model_instance` has no parameter to express).
- [x] 4.4 Migrated the remaining ~13 sites to `Runtime::create_model_instance` via a new `create_instance(&mut Runtime) -> ModelInstanceId` contract_tests helper; redefined `definition()` to build via `Runtime::create_model_instance` + `.definition().clone()` (still returns an owned, field-mutable `ModelInstanceDefinition` for the one remaining non-`.create()` consumer, `sharing_policy_considers_tenant_adapter_cache_privacy_and_affinity`); added `trusted_runtime()`/`loaded_context(&mut Runtime)` helpers since `contract_tests` can no longer call `ModelLoadingCoordinator::load` or build a bare, untrusted `Runtime::initialize` and expect loading to succeed.

## 5. External-bypass regression tests

- [x] 5.1/5.2 Not added: on review, a crate-internal test calling a `pub(crate)` primitive directly cannot demonstrate anything a compile-time visibility seal doesn't already guarantee by construction (crate-internal code being able to call crate-internal functions was never in question). This matches the established precedent already recorded in `CHANGELOG.md` for round 9's equivalent field-sealing fixes ("no new runtime tests, matching the compile-time-guarantee precedent round 6's `LoadedModelContext` sealing already established"). The behavioral guarantees that *do* matter -- the sealed facades actually enforcing trust/provenance -- are already covered by round 10's `inference_api_load_model_rejects_when_runtime_trust_store_does_not_trust_digest` and `runtime_create_model_instance_rejects_architecture_disagreeing_with_resolved_plan`, both still passing unchanged.

## 6. Verification

- [x] 6.1 `cargo build --locked --workspace --all-targets --all-features` clean.
- [x] 6.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` clean.
- [x] 6.3 `cargo fmt --all -- --check` clean.
- [x] 6.4 `cargo test --locked --workspace --all-targets --all-features`: 64 + 1182 (+11 relocated) + 173 (-11 relocated) passed, 0 failed -- net coverage unchanged, confirming the relocation moved tests rather than losing them.
- [x] 6.5 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean after fixing one private-intra-doc-link (`load_model`'s doc comment linked to the now-private `ModelLoadingCoordinator::load`; changed to plain-text reference).
- [x] 6.6 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean (same pre-existing, unrelated warnings as prior rounds).
- [x] 6.7 Coverage ratchet: 79.01% local (above the 78.89% accepted baseline).
- [x] 6.8 `openspec validate --all --strict`: 76/76.
- [x] 6.9 Live `magnetar run qwen-test "Hello"` unaffected: `[generated token ids: 239 239 239 239]`.
- [x] 6.10 Pushed as `0b353fe`; CI run 33910944204 confirmed green (`status: completed`, `conclusion: success`, 0 non-successful jobs) via direct `gh run view --json`.

## 7. Close out

- [x] 7.1 Diffed all three touched specs (order-independent requirement-heading comparison plus direct reads of both modified requirement bodies): each gained exactly the one new requirement authored for this Change; both MODIFIED bodies merged with all scenarios intact. No content dropped.
- [x] 7.2 Archived as `2026-09-04-seal-model-loading-and-instance-creation-primitives`. `openspec validate --all --strict`: 75/75.
- [x] 7.3 `CHANGELOG.md` updated: round-11's two P0s closed, Provider/Device gap honestly documented as a limitation; Architecture Freeze #1 status updated.
