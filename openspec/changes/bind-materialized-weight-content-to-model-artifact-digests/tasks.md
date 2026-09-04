## 1. Spec

- [x] 1.1 `model-artifact`: new "Tensor Content Digest Binding" requirement.
- [x] 1.2 `model-loading`: "Weight Materialization Sources Real Artifact Bytes" strengthened with content-digest verification.
- [x] 1.3 `model-instance`: new "Materialized Weight Content Matches Its Declared Digest When One Exists" requirement. `openspec validate --strict bind-materialized-weight-content-to-model-artifact-digests` passes.

## 2. `ModelTensorMetadata` gains a digest field

- [x] 2.1 Added `pub digest: Option<ModelDigest>` to `ModelTensorMetadata` (`model.rs`).
- [x] 2.2 `RawTensor` (the YAML-deserialization intermediate struct) gained `#[serde(default)] digest: Option<String>`; `TryFrom<RawTensor>` threads it via `raw.digest.map(ModelDigest::parse).transpose()?`, matching the existing pattern used for the artifact-level digest elsewhere in the same file.
- [x] 2.3 `cargo build` clean. 10 pre-existing `ModelTensorMetadata` struct-literal sites across the crate (roadmap types, test fixtures) updated with `digest: None`; none of them are YAML-parsed, so the "existing manifest without per-tensor digests still parses with `digest: None`" property is exercised directly by every existing manifest-parsing test continuing to pass unmodified (no dedicated new test needed for this specific sub-point).

## 3. Canonical `HostTensor` content-digest helper

- [x] 3.1 Added `HostTensor::content_bytes(&self) -> Vec<u8>` (`reference_cpu.rs`, next to `HostTensor` itself) -- little-endian `f32` concatenation. Placed on `HostTensor` as a method rather than a standalone function so `reference_cpu.rs` needs no new dependency on `model.rs`'s `ModelDigest` type; callers combine `tensor.content_bytes()` with `ModelDigest::sha256`/`verify_bytes` themselves. `ModelDigest::sha256`/`verify_bytes` already existed and needed no changes.
- [x] 3.2 Covered indirectly: the mismatch/matching-content e2e tests (group 7) exercise both directions (different data -> different digest -> rejected; same data -> same digest -> accepted) through the real entrypoint rather than a narrower unit test on the helper alone.

## 4. Fixture inventory carries real digests

- [x] 4.1 Added `e2e_fixture_weight_inventory_with_digests` (`first_native_runtime.rs`), layered on top of the existing `e2e_fixture_weight_inventory` + `e2e_fixture_weights_from_real_artifact` (real, already-parsed `HostTensor`s -- not `e2e_fixture_weights`'s synthetic in-memory values).
- [x] 4.2 `e2e_fixture_manifest` rewritten to build its tensor YAML from `e2e_fixture_weight_inventory_with_digests` directly (previously iterated `qwen_expected_tensor_names`/`qwen_expected_tensor_shape` separately); each tensor entry now emits a real `digest: sha256:...` line.
- [x] 4.3 Confirmed by the group-7 e2e tests and the live `qwen-test` run below -- both require the fixture's declared digests to actually match its real materialized bytes, which they do.

## 5. Thread `required_weight_digests` through Model Loading / Model Instance

- [x] 5.1 `LoadedModelContext` gained `pub(crate) required_weight_digests: BTreeMap<String, ModelDigest>`, populated in `ModelLoadingCoordinator::load()` from `manifest.tensors` (`filter_map` over entries with `Some(digest)`), mirroring `required_weight_names` exactly. No signature changes.
- [x] 5.2 `ModelInstanceDefinition` gained the matching `pub(crate)` field, carried through `from_loaded_context`.
- [x] 5.3 `cargo build` clean.

## 6. Transaction-level verification

- [x] 6.1 `WeightMaterializationTransaction::stage_weight` now checks `required_weight_digests` (fetched via `runtime.model_instance(instance)?.definition()`) before Memory Manager admission; on mismatch, returns `InferenceApiError::WeightContentDigestMismatch` without admitting or writing anything.
- [x] 6.2 Added `InferenceApiError::WeightContentDigestMismatch { reason: String }` with its own `Display` arm.
- [x] 6.3 Confirmed: the new check returns a plain `Result::Err` before any staging happens, so `materialize_model_instance_weights`'s existing loop (`transaction.abort(runtime)`, transition to `Failed`) handles it identically to any other `stage_weight` failure -- no special-casing needed, verified by the group-7 mismatch test observing exactly that path.

## 7. Tests

- [x] 7.1 Content mismatch rejected: `check_materialize_model_instance_weights_rejects_content_digest_mismatch` (+ `e2e_materialize_model_instance_weights_rejects_content_digest_mismatch` `#[test]` wrapper) -- tampers one real fixture tensor's bytes, calls `materialize_model_instance_weights` directly via `load_fixture_instance_with_weights`, asserts the error is specifically `WeightContentDigestMismatch` (matched via its `Display` text), not merely some error.
- [x] 7.2 Matching content accepted: `check_materialize_model_instance_weights_accepts_matching_content` (+ wrapper) -- real fixture bytes via `e2e_fixture_weights_from_real_artifact`, materializes successfully through the same entrypoint (regression guard for the happy path).
- [x] 7.3 No-digest-declared tensor is permissive: not given a dedicated new test -- every other existing test in the suite (1163 of the 1165 lib tests) uses generic fixtures/definitions with `required_weight_digests` empty and continues to materialize normally, which is a stronger regression guard than one isolated test would be.
- [x] 7.4 Live `magnetar run qwen-test "Hello"` confirmed unaffected: `[generated token ids: 239 239 239 239]`, identical to every prior round.
- [x] 7.5 (not originally planned, added during implementation) `check_weight_byte_change_alters_generated_logits` -- a *pre-existing* test proving mutated weight bytes are numerically consumed, not ignored -- needed a digest-free clone of the fixture manifest to keep working, since its own mutated bytes would otherwise (correctly) now be rejected by this Change's new check. Fixed by cloning `fixture` and stripping `.manifest.tensors[*].digest` before use, isolating that test's own orthogonal concern (bytes are consumed) from this Change's concern (bytes must match their declared digest).

## 8. Verification

- [x] 8.1 `cargo build --locked --workspace --all-targets --all-features` clean.
- [x] 8.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` clean.
- [x] 8.3 `cargo fmt --all -- --check` clean (one diff found and applied: a `format!` call collapsed onto one line).
- [x] 8.4 `cargo test --locked --workspace --all-targets --all-features`: 1165 lib tests (+2 from this Change) + 184 `contract_tests`, 0 failed.
- [x] 8.5 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean (no private-intra-doc-link issue this round).
- [x] 8.6 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean.
- [x] 8.7 Coverage ratchet: 79.00% (above the 78.89% baseline).
- [x] 8.8 `openspec validate --all --strict`: 77/77 (76 canonical + this active Change).
- [x] 8.9 Live `magnetar run qwen-test "Hello"` unaffected.
- [ ] 8.10 Push, confirm a full green CI run via `gh run view --json status,conclusion` and the jobs list directly (never a piped `gh run watch`).

## 9. Close out

- [ ] 9.1 Diff the canonical spec files this Change touches before and after archiving, to confirm the archive-merge did not silently drop anything (the recurring check this session established after `9939232`'s regression).
- [ ] 9.2 Archive this Change.
- [ ] 9.3 Update `CHANGELOG.md`: P0-2 closed; update Architecture Freeze #1 status (should move from CANDIDATE back toward accepted if no other blocker remains open).
