## 1. Spec

- [x] 1.1 `model-instance`: "Model Instance Readiness" and "Model Instance Warmup" gain requirements that the public surface cannot reach `Ready` without Runtime-verified evidence, and that `weights_materialized`/`provider_ready` consult real backing state. `openspec validate --strict seal-model-instance-readiness-authority` passes.

## 2. Seal the direct bypass (P0-A)

- [x] 2.1 `ModelInstance::transition_to`, `ModelInstance::mark_ready`, `ModelInstance::warmup` become `pub(crate)`.
- [x] 2.2 `ModelInstanceManager::mark_ready`, `ModelInstanceManager::warmup` become `pub(crate)`.
- [x] 2.3 `ModelInstance.lifecycle`/`.readiness` fields become `pub(crate)` (not fully private -- this crate's own test suite needs them for defense-in-depth tests a fully private field would make impossible to write from a sibling module). Public read-only accessors `lifecycle()`/`readiness()` added.
- [x] 2.4 Verified `magnetar-cli` calls none of the sealed methods directly (`grep -rn "mark_ready\|\.transition_to(\|\.warmup(" magnetar-cli/src/` -- no matches) before sealing, confirming no real embedder breakage.
- [x] 2.5 `warm_model_instance` (`inference_api.rs`) now routes through `ModelInstanceManager::warmup` (not `ModelInstance::warmup` directly), fixing a pre-existing observability gap discovered in the process: the Manager-level wrapper emits `ModelInstanceObservationKind::Ready`/`Failed`, which the direct-to-`ModelInstance` call from round 1 had been silently skipping.

## 3. Strengthen weights_materialized (P0-B)

- [x] 3.1 `derive_effective_readiness_checks` now also requires every `resource_bindings.weights` entry to have a matching `MemoryManager::tensor_residency` record, not just a non-empty map.

## 4. Strengthen provider_ready (P0-C)

- [x] 4.1 `derive_effective_readiness_checks` now also requires `Provider::status_snapshot().accepts_new_work_by_default()`, not just `execution_api().is_some()`.

## 5. Migrate tests off the sealed primitives

- [x] 5.1 `magnetar-runtime/tests/contract_tests/model_instance.rs`: added `bind_fake_weight`/`reach_ready` helpers using real `MemoryManager` admission + `record_tensor_residency` + `warm_model_instance` -- the same contract a real embedder must use, not a parallel test-only path. Migrated all ~14 affected tests (bare `ModelInstanceManager` → `Runtime`, raw `mark_ready`/`transition_to`/`warmup` calls → `reach_ready`/business methods/`warm_model_instance`, raw `.lifecycle`/`.readiness` field reads → `.lifecycle()`/`.readiness()`).
- [x] 5.2 `provider_and_device_status_drive_instance_lifecycle` split into three separately-`Runtime`-backed instances (one per scenario) instead of one instance with a manually-reset `readiness` field, since direct field reset from outside the crate is no longer available (and was already a slightly awkward test structure).
- [x] 5.3 `magnetar-runtime/src/tests.rs`: fixed `inference_api_warm_model_instance_reaches_ready_when_weights_and_provider_are_real` (from round 1) -- it had inserted a bare `TensorResourceId` with no residency, exactly the gap task 3.1 now closes; it was failing for the *right* reason once the fix landed, corrected to bind a real residency-backed weight.
- [x] 5.4 Added `inference_api_warm_model_instance_rejects_weight_binding_without_residency` (audit test 23.3) and `inference_api_warm_model_instance_rejects_provider_that_rejects_new_work` (audit test 23.5, using the existing `TestProvider`/`TestProviderExecutionApi` test doubles with a `Saturated` status snapshot).

## 6. Verification

- [x] 6.1 `cargo build -p magnetar-runtime --lib --tests` clean (iterated through 48 compile errors from the visibility changes down to 0, all in the two affected test files).
- [x] 6.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`, `cargo fmt --all --check`, `CARGO_INCREMENTAL=0`. Both clean (one unused-import and one formatting fix needed after the test migration).
- [x] 6.3 `cargo test --locked --workspace --all-targets --all-features`: 64 + 1153 + 182 passed, 0 failed (iterated through 2 real test bugs the migration surfaced: an ID collision between a fixture's hardcoded `MemoryAllocationId::new(1)` and the real allocator's first-issued id, and a Provider-pinned fixture that needed its fake placement injected after reaching Ready rather than before, since no such Provider was ever registered).
- [x] 6.4 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean.
- [x] 6.5 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean.
- [x] 6.6 Coverage ratchet: 78.91%, above the 78.89% baseline.
- [x] 6.7 `openspec validate --all --strict`: 77/77 passed.
- [x] 6.8 Live `magnetar run qwen-test "Hello"` unaffected (production path never calls `warm_model_instance` directly; confirmed by inspection and by the live run producing the same output as before this Change).
- [ ] 6.9 Push, confirm a full green CI run via `gh run view --json status,conclusion` and the jobs list directly (never a piped `gh run watch`).

## 7. Close out

- [ ] 7.1 Archive this Change.
- [ ] 7.2 Update `CHANGELOG.md`'s Architecture Freeze #1 note per the revalidation audit's own criterion.
- [ ] 7.3 File/extend the deferred P1 issues the audit re-confirmed as still open (`#41` extension for `release_tensor`'s missing structured error channel; `#38`/`#39`/`#40` unchanged) -- these were not part of this Change's scope and remain independently tracked.
