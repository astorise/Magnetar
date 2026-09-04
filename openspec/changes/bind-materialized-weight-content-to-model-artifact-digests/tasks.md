## 1. Spec

- [x] 1.1 `model-artifact`: new "Tensor Content Digest Binding" requirement.
- [x] 1.2 `model-loading`: "Weight Materialization Sources Real Artifact Bytes" strengthened with content-digest verification.
- [x] 1.3 `model-instance`: new "Materialized Weight Content Matches Its Declared Digest When One Exists" requirement. `openspec validate --strict bind-materialized-weight-content-to-model-artifact-digests` passes.

## 2. `ModelTensorMetadata` gains a digest field

- [ ] 2.1 Add `pub digest: Option<ModelDigest>` to `ModelTensorMetadata` (`model.rs`).
- [ ] 2.2 Find the YAML-deserialization intermediate struct for manifest `tensors:` entries and add the matching optional field (`#[serde(default)]` so existing manifest YAML without a per-tensor `digest:` key keeps parsing unchanged); thread it through whatever conversion produces the real `ModelTensorMetadata` (parsing the string via `ModelDigest::parse` if the intermediate field is a raw string, matching how the artifact-level digest is already parsed elsewhere in this file).
- [ ] 2.3 `cargo build -p magnetar-runtime --lib` clean. Confirm via a quick existing-manifest-parsing test that a manifest without per-tensor digests still parses with `digest: None` on every tensor.

## 3. Canonical `HostTensor` content-digest helper

- [ ] 3.1 Add a shared function computing a `ModelDigest` from a `HostTensor`'s canonical byte representation (little-endian `f32` concatenation, the same representation `e2e_fixture_weight_digest`'s existing per-tensor hashing already uses) -- exact placement (`reference_cpu.rs` near `HostTensor`, or `first_native_runtime.rs` near `e2e_fixture_weight_digest`) decided by whichever avoids a new cross-module dependency.
- [ ] 3.2 Unit test: two `HostTensor`s with identical `shape`/`data` produce the same digest; different `data` produces a different digest.

## 4. Fixture inventory carries real digests

- [ ] 4.1 Populate real per-tensor digests on the E2E fixture's tensor inventory, computed from `e2e_fixture_weights_from_real_artifact`'s already-real, already-parsed `HostTensor`s (not from `e2e_fixture_weights`'s synthetic in-memory values) via the group-3 helper.
- [ ] 4.2 `e2e_fixture_manifest`'s YAML emits the digest per tensor entry.
- [ ] 4.3 `cargo build`/existing fixture-construction tests confirm the fixture manifest now carries real, non-`None` digests for every tensor, and that they actually verify against the real file's bytes (a direct assertion, not just "it parses").

## 5. Thread `required_weight_digests` through Model Loading / Model Instance

- [ ] 5.1 `LoadedModelContext` gains `pub(crate) required_weight_digests: BTreeMap<String, ModelDigest>`, populated in `ModelLoadingCoordinator::load()` from `manifest.tensors` (only entries with `Some(digest)`), mirroring `required_weight_names`'s exact threading shape -- no signature changes to `load()`, `create_model_instance`, or `from_loaded_context`.
- [ ] 5.2 `ModelInstanceDefinition` gains the matching `pub(crate)` field, carried through `from_loaded_context`.
- [ ] 5.3 `cargo build` clean.

## 6. Transaction-level verification

- [ ] 6.1 `WeightMaterializationTransaction::stage_weight` (`first_native_runtime.rs`): before Memory Manager admission, if the instance's `required_weight_digests` has an entry for the tensor name being staged, compute the group-3 helper's digest over the supplied `HostTensor` and call `ModelDigest::verify_bytes`-equivalent comparison; on mismatch, return a new, dedicated error (do not reuse `MemoryAdmissionFailed`) and do not admit or write anything for that tensor.
- [ ] 6.2 New `InferenceApiError` variant for a content-digest mismatch, with a clear message naming the tensor.
- [ ] 6.3 Confirm the existing abort/rollback path (`materialize_model_instance_weights`'s caller loop, already calling `transaction.abort(runtime)` and transitioning the instance to `Failed` on any `stage_weight` error) handles this new error variant the same way as any other staging failure -- no special-casing needed if the error surfaces through the same `Result::Err` path.

## 7. Tests

- [ ] 7.1 Content mismatch rejected: real fixture manifest (digests populated), caller supplies tampered bytes under a correct tensor name via `materialize_model_instance_weights` -- rejected, no evidence minted, instance not Ready.
- [ ] 7.2 Matching content accepted: real fixture manifest, real fixture bytes -- materializes and reaches Ready exactly as before (regression guard for the existing happy path).
- [ ] 7.3 No-digest-declared tensor is permissive: a manifest/definition with `required_weight_digests` empty (or missing an entry for one tensor) -- that tensor's materialization is not blocked on content-digest grounds, proving this Change does not regress any manifest that predates it.
- [ ] 7.4 Live `magnetar run qwen-test "Hello"` still produces the same output -- the real fixture's real bytes match its own real digests, so the production path is unaffected.

## 8. Verification

- [ ] 8.1 `cargo build --locked --workspace --all-targets --all-features` clean.
- [ ] 8.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` clean.
- [ ] 8.3 `cargo fmt --all -- --check` clean.
- [ ] 8.4 `cargo test --locked --workspace --all-targets --all-features`: full suite passing, count recorded here.
- [ ] 8.5 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean.
- [ ] 8.6 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean.
- [ ] 8.7 Coverage ratchet: at or above baseline (no lowering the baseline to pass).
- [ ] 8.8 `openspec validate --all --strict` passing, item count recorded here.
- [ ] 8.9 Live `magnetar run qwen-test "Hello"` unaffected.
- [ ] 8.10 Push, confirm a full green CI run via `gh run view --json status,conclusion` and the jobs list directly (never a piped `gh run watch`).

## 9. Close out

- [ ] 9.1 Diff the canonical spec files this Change touches before and after archiving, to confirm the archive-merge did not silently drop anything (the recurring check this session established after `9939232`'s regression).
- [ ] 9.2 Archive this Change.
- [ ] 9.3 Update `CHANGELOG.md`: P0-2 closed; update Architecture Freeze #1 status (should move from CANDIDATE back toward accepted if no other blocker remains open).
