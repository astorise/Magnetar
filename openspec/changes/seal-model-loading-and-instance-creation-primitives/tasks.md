## 1. Spec

- [ ] 1.1 `model-loading`: new "Model Loading Is Reachable Only Through the Runtime-Sealed Loading API" requirement.
- [ ] 1.2 `model-instance`: "Model Instance References Architecture Implementation" corrected to describe the unresolved-Provider/Device case as a documented limitation, not a caller-authority grant; new "Model Instance Creation Is Reachable Only Through Runtime" requirement.
- [ ] 1.3 `inference-api`: "Provider Preferences Are Non-Authoritative" gains a cross-reference to the same documented limitation. `openspec validate --strict seal-model-loading-and-instance-creation-primitives` passes.

## 2. Confirm no non-test caller depends on either primitive

- [ ] 2.1 Grep `magnetar-cli`, `formats/gguf`, `formats/safetensors` for `ModelLoadingCoordinator::load`, `ModelInstanceDefinition::from_loaded_context`, `ModelInstanceManager::create` / `model_instances_mut().create` -- confirm zero non-test hits before sealing (done during design; re-confirm at implementation time in case anything changed).

## 3. Seal `ModelLoadingCoordinator::load`

- [ ] 3.1 `pub fn load` -> `pub(crate) fn load` (`model_loading.rs`).
- [ ] 3.2 `cargo build`/`cargo test` enumerate every break.
- [ ] 3.3 Relocate `contract_tests/model_loading.rs`'s tests that exercise `ModelLoadingCoordinator::load`'s own contract (untrusted rejection, memory-budget/quantization/allocation failure mapping, Ready-context shape) into `magnetar-runtime/src/tests.rs`.
- [ ] 3.4 Whatever remains in `contract_tests/model_loading.rs` after relocation (if anything) is genuinely testing public, non-`load`-dependent surface (e.g. `resolve_architecture`, `ModelLoadingState::can_transition_to`, the standalone `compute_dtype_supported`/`invalidates_kv_cache_on_unload`/`reload_is_new_loading_process` helpers) and stays there unchanged.

## 4. Seal `ModelInstanceDefinition::from_loaded_context` and `ModelInstanceManager::create`

- [ ] 4.1 Both `pub fn` -> `pub(crate) fn` (`model_instance.rs`).
- [ ] 4.2 `cargo build`/`cargo test` enumerate every break.
- [ ] 4.3 Relocate `contract_tests/model_instance.rs`'s `cloned_definition_does_not_inherit_weight_authority` and `reload_replacement_does_not_inherit_original_weight_authority` into `magnetar-runtime/src/tests.rs` (they test `.create()`'s own reset-on-create guarantee against a hand-cloned definition, which has no `Runtime::create_model_instance` equivalent).
- [ ] 4.4 Migrate every other `contract_tests/model_instance.rs` site using `definition()` + `runtime.model_instances_mut().create(...)` to `runtime.create_model_instance(&loaded_context(), implementation(), ResourceAffinity::new(FallbackClass::Transparent))` in place.

## 5. External-bypass regression tests

- [ ] 5.1 `magnetar-runtime/src/tests.rs`: a test demonstrating that a `Runtime` sealed to not trust a digest cannot have that artifact loaded into it even via crate-internal direct use of `ModelLoadingCoordinator::load` with a separately-evaluated, legitimately-obtained `Trusted` decision -- the closest an in-crate test can get to proving no external crate can do this anymore (the compile-time seal itself is the real proof; this test documents the intent).
- [ ] 5.2 `magnetar-runtime/src/tests.rs`: equivalent for the instance-creation bypass -- a definition built via `from_loaded_context` with a disagreeing architecture, registered via `ModelInstanceManager::create` directly (crate-internal), still produces the disagreeing instance (proving the cross-check genuinely lives only in `Runtime::create_model_instance`, not duplicated lower down) -- documents why the seal, not a lower-level check, is what closes this.

## 6. Verification

- [ ] 6.1 `cargo build --locked --workspace --all-targets --all-features` clean.
- [ ] 6.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` clean.
- [ ] 6.3 `cargo fmt --all -- --check` clean.
- [ ] 6.4 `cargo test --locked --workspace --all-targets --all-features`: 0 failed.
- [ ] 6.5 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean.
- [ ] 6.6 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean.
- [ ] 6.7 Coverage ratchet: does not drop below the current accepted baseline.
- [ ] 6.8 `openspec validate --all --strict` clean.
- [ ] 6.9 Live `magnetar run qwen-test "Hello"` unaffected.
- [ ] 6.10 Push, confirm a full green CI run via `gh run view --json status,conclusion` and the jobs list directly (never a piped `gh run watch`).

## 7. Close out

- [ ] 7.1 Diff the canonical spec files this Change touches before and after archiving, to confirm the archive-merge did not silently drop anything.
- [ ] 7.2 Archive this Change.
- [ ] 7.3 Update `CHANGELOG.md`: round-11's findings closed (or, for the Provider/Device gap, honestly documented); Architecture Freeze #1 status updated.
