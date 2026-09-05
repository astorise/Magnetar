## 1. Spec

- [x] 1.1 `model`: "Model Artifact Trust" strengthened with Runtime-sourced, non-caller-substitutable trust evaluation.
- [x] 1.2 `inference-api`: "Model Loading API" strengthened to require the trust decision comes from the performing Runtime, not a call parameter.
- [x] 1.3 `runtime`: new "Runtime-Sealed Trust Configuration" requirement.
- [x] 1.4 `model-instance`: "Model Instance References Architecture Implementation" strengthened with the plan cross-check; new "Materialized Weight Content Matches Its Declared Shape And Storage Dtype" requirement. `openspec validate --strict seal-runtime-model-trust-and-provenance-authority` passes.

## 2. Trust: Runtime-owned, sealed configuration

- [x] 2.1 `RuntimeBuilder` gains `pub fn trust_store(mut self, store: ModelTrustStore) -> Self`; `Runtime` gains a private `trust: ModelTrustStore` field defaulting to `ModelTrustStore::default()` when unset, and `pub(crate) fn trust_store(&self) -> &ModelTrustStore` (no public getter that hands back an owned/mutable copy).
- [x] 2.2 `load_model`/`load_model_observed` (`inference_api.rs`) take `runtime: &mut Runtime` instead of a caller-supplied `trust: &ModelTrustDecision` (revised from the planned `&Runtime`: `memory` is no longer a separate parameter either -- `runtime.memory_mut()` and `runtime.trust_store()` as two separate borrows of the same `Runtime` in one call is a real aliasing conflict, so `load_model` now derives both internally from one `&mut Runtime`).
- [x] 2.3 `cargo build` enumerated every call site: `magnetar-cli/src/commands.rs` (`run_load_model`), `first_native_runtime.rs` (4 sites -- one more than the initial 3-site estimate, `check_weight_materialization_failure_never_reaches_ready`'s own inline `Runtime::builder()`), `tests.rs` (3 sites), plus ~25 fixture-Runtime-construction helper call sites needing a trust-configured Runtime (see 2.4).
- [x] 2.4 `first_native_runtime.rs`'s `build_runtime()` (used by 2 fixture-less tests with nothing to trust) kept as-is; new `build_runtime_trusting_fixture(fixture)` added and substituted at its ~23 real call sites, plus `.trust_store(...)` added directly to `build_runtime_with_model_execution_engine(_and_forced_token)` and one bespoke inline `Runtime::builder()` chain. 3 additional untrusted-Runtime sites surfaced only by `cargo test` (not `cargo build`, since they don't touch signatures) in `first_native_runtime/tests.rs` and `first_native_runtime.rs` itself, fixed the same way.
- [x] 2.5 `cargo build --workspace --all-targets --all-features` clean.

## 3. Instance provenance: cross-check against the resolved plan

- [x] 3.1 New `ModelInstanceError` variants: `ArchitectureMismatch { expected: ModelArchitecture, actual: ModelArchitecture }` and `AffinityMismatch { reason: String }`, each with a `Display` arm following this crate's existing convention.
- [x] 3.2 `Runtime::create_model_instance` checks, before constructing the definition: `architecture.architecture == loaded.plan().architecture` (reject on mismatch); when `loaded.plan().provider_binding`/`.device_binding` is `Some`, `affinity.provider()`/`.device()` must agree (reject on mismatch); an unresolved plan field imposes no constraint.
- [x] 3.3 `cargo build`/`cargo test` clean; no existing call site (fixtures, contract tests) was relying on architecture/affinity silently disagreeing with the plan -- confirmed by inspection before implementing (`contract_tests::model_instance`'s 4 `create_model_instance` call sites all use `implementation()`/an affinity matching the same manifest's own architecture, with `plan.provider_binding`/`.device_binding` never resolved to `Some` anywhere in the real loading pipeline today).

## 4. Weight provenance: shape/dtype cross-check

- [x] 4.1 `LoadedModelContext`/`ModelInstanceDefinition` gain `pub(crate) required_weight_shapes: BTreeMap<String, (Vec<u64>, ModelDType)>`, populated in `ModelLoadingCoordinator::load()` from `manifest.tensors`, mirroring `required_weight_digests` exactly.
- [x] 4.2 New `InferenceApiError::WeightShapeOrDtypeMismatch { reason: String }` with its own `Display` arm.
- [x] 4.3 `WeightMaterializationTransaction::stage_weight` checks `required_weight_shapes.get(name)` before the existing digest check: rejects if `tensor.shape` disagrees with the declared shape, or if the declared `storage_dtype` is not `F32`, regardless of digest presence.
- [x] 4.4 `cargo test` surfaced one real fixture inconsistency this check exposed (not merely "confirmed already matching" as anticipated): `contract_tests::model_instance`'s generic fixture manifest declared `transformer.wte.weight` as `bf16`/`[4, 8]` while its `bind_fake_weight` helper supplied fabricated F32 `[1]`-shaped content -- exactly the class of gap this Change closes. Fixed by aligning the fixture's declared dtype to `f32` and the helper's supplied shape to `[4, 8]`, both now real (if fake-valued) F32 content consistent with what it declares. The real E2E fixture (`first_native_runtime.rs`, real parsed artifact bytes) needed no changes -- confirmed by the full suite passing unmodified there.

## 5. Tests

- [x] 5.1 Trust: `inference_api_load_model_rejects_when_runtime_trust_store_does_not_trust_digest` (`tests.rs`) -- a `Runtime` built with the default (nothing-trusted) store rejects loading a manifest, asserting on `ModelArtifactUntrusted` in the resulting error text.
- [x] 5.2 Provenance (architecture): `runtime_create_model_instance_rejects_architecture_disagreeing_with_resolved_plan` -- asserts `ModelInstanceError::ArchitectureMismatch` on disagreement and success on agreement.
- [x] 5.3 Provenance (affinity): `runtime_create_model_instance_rejects_affinity_disagreeing_with_resolved_provider_binding` -- directly mutates `loaded.plan.provider_binding` (accessible in-crate, `pub(crate)`) to force the "loading phase resolved a provider" scenario design.md flagged as otherwise unreachable through today's real pipeline; asserts `ModelInstanceError::AffinityMismatch` on disagreement and success on agreement. The "unresolved plan field imposes no constraint" branch is already exercised by every other passing test using `ResourceAffinity::new(FallbackClass::Transparent)` with no provider/device set.
- [x] 5.4 Weight shape/dtype: `materialize_model_instance_weights_rejects_shape_mismatch`, `materialize_model_instance_weights_rejects_non_f32_declared_dtype`, `materialize_model_instance_weights_accepts_matching_shape_and_dtype` (all `tests.rs`, using a new `manifest_with_one_tensor` helper).
- [x] 5.5 Live `magnetar run qwen-test "Hello"` unaffected: `[generated token ids: 239 239 239 239]`.

## 6. Verification

- [x] 6.1 `cargo build --locked --workspace --all-targets --all-features` clean.
- [x] 6.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` clean.
- [x] 6.3 `cargo fmt --all -- --check` clean.
- [x] 6.4 `cargo test --locked --workspace --all-targets --all-features`: 64 + 1171 (+6 from this Change) + 184 passed, 0 failed.
- [x] 6.5 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean.
- [x] 6.6 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean (pre-existing, unrelated `unused variable` warnings under this target only, not caused by this Change).
- [x] 6.7 Coverage ratchet: 79.00% local (above the 78.89% accepted baseline).
- [x] 6.8 `openspec validate --all --strict`: 76/76.
- [x] 6.9 Live `magnetar run qwen-test "Hello"` unaffected: `[generated token ids: 239 239 239 239]`, identical to every prior round.
- [x] 6.10 Pushed as `045a536`; CI run 33879062757 confirmed green (`status: completed`, `conclusion: success`, 0 non-successful jobs) via direct `gh run view --json`.

## 7. Close out

- [x] 7.1 Diffed all four touched specs (order-independent requirement-heading comparison plus direct reads of both new/modified requirement bodies): `model` and `inference-api` unchanged requirement sets with the two MODIFIED bodies merged correctly (including both new scenarios each); `runtime` and `model-instance` each gained exactly the one new requirement authored for this Change. No content dropped.
- [x] 7.2 Archived as `2026-09-04-seal-runtime-model-trust-and-provenance-authority`. `openspec validate --all --strict`: 75/75.
- [x] 7.3 `CHANGELOG.md` updated: round-10's P0-A/P0-B/P0-C closed; Architecture Freeze #1 status updated.
