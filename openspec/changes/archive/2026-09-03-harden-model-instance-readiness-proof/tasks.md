## 1. Spec

- [x] 1.1 `model-instance`: new "Model Instance Resume Revalidates Readiness" requirement; "Model Instance Readiness" strengthened for inventory completeness and real Provider-backed evidence. `openspec validate --strict harden-model-instance-readiness-proof` passes.
- [x] 1.2 Confirmed `model-loading`'s "Qwen Loading Validates Tensor Inventory" and "Partial Loading Policy" requirements already existed and needed no wording change -- this Change is the implementation catching up, not a new decision.

## 2. Resume revalidation (audit P0-1)

- [x] 2.1 `ModelInstance::resume()` now only transitions `Suspended -> Loading`, not all the way to `Ready`.
- [x] 2.2 `resume_model_instance` (`inference_api.rs`) completes the transition through `warm_model_instance`, reusing the same Runtime-derived-evidence path every other route to `Ready` uses.
- [x] 2.3 `ModelInstance::resume` stays `pub` (not sealed like `mark_ready`/`transition_to`/`warmup`): since it no longer reaches `Ready` by itself, calling it directly is harmless.

## 3. Weight materialization proof (audit P0-2)

- [x] 3.1 `LoadedModelContext` gains `required_weight_names: BTreeSet<String>`, populated by `ModelLoadingCoordinator::load()` from `manifest.tensors`.
- [x] 3.2 `ModelInstanceDefinition` gains a `pub(crate)` field of the same name, carried through `from_loaded_context`. Deliberately `pub(crate)`, not `pub`, and empty means "unknown" (falls back to the prior heuristic), not "nothing required" -- see design.md.
- [x] 3.3 `derive_effective_readiness_checks`: `weights_materialized` now also requires every `required_weight_names` entry (when non-empty) to be a bound key, and resolves each bound weight's residency to its recorded Provider and calls `read_tensor` to confirm real backing storage, not just a residency record.
- [x] 3.4 Considered and rejected a `WeightMaterializationState` state machine (the audit's own "Option recommandée") in favor of the `read_tensor`-based check -- see design.md's Non-Goals for why.

## 4. Tests

- [x] 4.1 `inference_api_resume_model_instance_revalidates_and_rejects_stale_evidence`: reach real Ready, suspend, invalidate the weight evidence during suspension (remove its `TensorResidency`), resume must fail and the instance must not be Ready.
- [x] 4.2 `inference_api_warm_model_instance_rejects_incomplete_mandatory_weight_inventory`: `required_weight_names` declares two tensors, only one is bound (with full real evidence) -- must not reach Ready.
- [x] 4.3 `inference_api_warm_model_instance_rejects_residency_without_provider_write`: a real Memory Manager allocation and a real `record_tensor_residency` call, but no `write_tensor` ever ran -- must not reach Ready.
- [x] 4.4 `magnetar-runtime/tests/contract_tests/model_instance.rs`'s `bind_fake_weight` helper updated: registers a `ReferenceCpuProvider` if not already present, performs a real `write_tensor` call, and binds under the fixture manifest's actual declared tensor name (`transformer.wte.weight`) rather than a generic key -- required once `required_weight_names` started being populated from that fixture's manifest. All ~13 tests using it fixed as a result, no test-specific changes needed beyond the shared helper.
- [x] 4.5 Both round-1/round-2 happy-path tests in `tests.rs` (`inference_api_model_instance_suspend_resume_drain_through_api_boundary`, `inference_api_warm_model_instance_reaches_ready_when_weights_and_provider_are_real`) fixed to supply real Provider-backed evidence -- they had been relying on the exact gaps this Change closes (bare `mark_ready`/a residency with no real Provider write) and were failing for the *correct* new reason once the fix landed. Extracted a shared `reach_ready_with_real_weight` helper to avoid duplicating the real-evidence setup a third time.

## 5. Verification

- [x] 5.1 `cargo build -p magnetar-runtime --lib --tests` clean.
- [x] 5.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`, `cargo fmt --all --check`, `CARGO_INCREMENTAL=0`. Both clean.
- [x] 5.3 `cargo test --locked --workspace --all-targets --all-features`: 64 + 1156 + 182 passed, 0 failed.
- [x] 5.4 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean.
- [x] 5.5 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean.
- [x] 5.6 Coverage ratchet: 78.93%, above the 78.89% baseline.
- [x] 5.7 `openspec validate --all --strict`: 77/77 passed.
- [x] 5.8 Live `magnetar run qwen-test "Hello"` unaffected (the production path uses `WeightMaterializationTransaction::commit` -> `mark_ready` directly, never `warm_model_instance`/`resume_model_instance`, so none of this Change's derivation changes apply to it; confirmed by the live run producing the same output).
- [ ] 5.9 Push, confirm a full green CI run via `gh run view --json status,conclusion` and the jobs list directly (never a piped `gh run watch`).

## 6. Close out

- [ ] 6.1 Archive this Change.
- [ ] 6.2 Update `CHANGELOG.md`'s Architecture Freeze #1 note per this audit round's own criterion.
