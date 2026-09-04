## 1. Spec

- [ ] 1.1 `model`: "Model Artifact Trust" strengthened with Runtime-sourced, non-caller-substitutable trust evaluation.
- [ ] 1.2 `inference-api`: "Model Loading API" strengthened to require the trust decision comes from the performing Runtime, not a call parameter.
- [ ] 1.3 `runtime`: new "Runtime-Sealed Trust Configuration" requirement.
- [ ] 1.4 `model-instance`: "Model Instance References Architecture Implementation" strengthened with the plan cross-check; new "Materialized Weight Content Matches Its Declared Shape And Storage Dtype" requirement. `openspec validate --strict seal-runtime-model-trust-and-provenance-authority` passes.

## 2. Trust: Runtime-owned, sealed configuration

- [ ] 2.1 `RuntimeBuilder` gains `pub fn trust_store(mut self, store: ModelTrustStore) -> Self`; `Runtime` gains a private `trust: ModelTrustStore` field defaulting to `ModelTrustStore::default()` when unset, and `pub(crate) fn trust_store(&self) -> &ModelTrustStore` (no public getter that hands back an owned/mutable copy).
- [ ] 2.2 `load_model`/`load_model_observed` (`inference_api.rs`) drop the `trust: &ModelTrustDecision` parameter and take `runtime: &Runtime` instead; internally call `runtime.trust_store().evaluate(manifest)`.
- [ ] 2.3 `cargo build` -- let the compiler enumerate every call site that breaks. Migrate each: `magnetar-cli/src/commands.rs` (`run_load_model`), `first_native_runtime.rs` (three sites, including the qwen-test live E2E fixture), `tests.rs` (three sites), and any `contract_tests` site the compiler surfaces that the earlier grep missed.
- [ ] 2.4 Test helpers that need an *untrusted* decision (e.g. an `untrusted()` fixture helper) migrate to configuring a `Runtime` whose trust store trusts a different digest than the manifest under test declares, rather than fabricating a decision inline.
- [ ] 2.5 `cargo build --workspace --all-targets --all-features` clean.

## 3. Instance provenance: cross-check against the resolved plan

- [ ] 3.1 New `ModelInstanceError` variants: `ArchitectureMismatch { expected: ModelArchitecture, actual: ModelArchitecture }` and `AffinityMismatch { reason: String }` (or equivalent typed fields), each with a `Display` arm following this crate's existing convention.
- [ ] 3.2 `Runtime::create_model_instance` checks, before constructing the definition: `architecture.architecture == loaded.plan().architecture` (reject on mismatch); when `loaded.plan().provider_binding`/`.device_binding` is `Some`, `affinity.provider()`/`.device()` must agree (reject on mismatch); an unresolved plan field imposes no constraint.
- [ ] 3.3 `cargo build` clean; confirm no existing call site (fixtures, contract tests) was relying on architecture/affinity silently disagreeing with the plan -- if one is, fix its fixture construction rather than loosening the check.

## 4. Weight provenance: shape/dtype cross-check

- [ ] 4.1 `LoadedModelContext`/`ModelInstanceDefinition` gain `pub(crate) required_weight_shapes: BTreeMap<String, (Vec<u64>, ModelDType)>`, populated in `ModelLoadingCoordinator::load()` from `manifest.tensors`, mirroring `required_weight_digests` exactly (same loop, same file, threaded through `from_loaded_context`).
- [ ] 4.2 New `InferenceApiError::WeightShapeOrDtypeMismatch { reason: String }` with its own `Display` arm.
- [ ] 4.3 `WeightMaterializationTransaction::stage_weight` checks `required_weight_shapes.get(name)` before the existing digest check: reject if `tensor.shape` disagrees with the declared shape, or if the declared `storage_dtype` is not `F32` (the only dtype this Runtime can materialize into `HostTensor` today) -- applies regardless of whether a digest exists for that tensor.
- [ ] 4.4 `cargo build` clean; confirm the e2e fixture's declared tensor shapes/dtypes actually match what it materializes (they should, since the fixture already round-trips real artifact bytes) -- no fixture changes anticipated, but verify rather than assume.

## 5. Tests

- [ ] 5.1 Trust: a test asserting a `Runtime` built with a trust store that does not trust a manifest's digest rejects loading through it, and that the free function accepts no caller-supplied override to bypass this.
- [ ] 5.2 Provenance (architecture): a test asserting `create_model_instance` rejects an `architecture` whose `ModelArchitecture` disagrees with `loaded.plan().architecture`, and accepts one that agrees.
- [ ] 5.3 Provenance (affinity): a test that forces a plan with a resolved `provider_binding`/`device_binding` (constructing the fixture's loading path so the plan actually resolves one, per design.md's noted risk that this may not happen by default) and asserts a disagreeing `affinity` is rejected, plus a companion test confirming an unresolved plan field imposes no constraint.
- [ ] 5.4 Weight shape/dtype: a test asserting a shape-mismatched tensor is rejected regardless of digest presence, a test asserting a tensor whose declared `storage_dtype` is non-F32 is rejected even with well-formed content, and a test asserting matching shape+dtype+digest still materializes normally (regression guard).
- [ ] 5.5 Live `magnetar run qwen-test "Hello"` unaffected.

## 6. Verification

- [ ] 6.1 `cargo build --locked --workspace --all-targets --all-features` clean.
- [ ] 6.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` clean.
- [ ] 6.3 `cargo fmt --all -- --check` clean.
- [ ] 6.4 `cargo test --locked --workspace --all-targets --all-features`: 0 failed.
- [ ] 6.5 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean.
- [ ] 6.6 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean.
- [ ] 6.7 Coverage ratchet: does not drop below the current accepted baseline.
- [ ] 6.8 `openspec validate --all --strict` clean.
- [ ] 6.9 Live `magnetar run qwen-test "Hello"` unaffected (re-confirm after the full verification pass, not just section 5's earlier check).
- [ ] 6.10 Push, confirm a full green CI run via `gh run view --json status,conclusion` and the jobs list directly (never a piped `gh run watch`).

## 7. Close out

- [ ] 7.1 Diff the canonical spec files this Change touches before and after archiving, to confirm the archive-merge did not silently drop anything.
- [ ] 7.2 Archive this Change.
- [ ] 7.3 Update `CHANGELOG.md`: round-10's three P0s closed; update Architecture Freeze #1 status.
