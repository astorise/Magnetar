## 1. Spec

- [x] 1.1 `model-instance`: "Model Instance Readiness", "Model Instance Warmup", "Generation Requires Ready Model Instance" gain explicit Runtime-derivation and lifecycle+readiness-consistency requirements. `openspec validate --strict runtime-owned-model-instance-readiness-authority` passes.

## 2. Runtime-derived readiness facts (item 1-3 of the auditor's DO NOW list)

- [x] 2.1 `ModelInstanceLifecycleState::supports_inference_use()` added (`model_instance.rs`): `Ready | Idle | Active`.
- [x] 2.2 `derive_effective_readiness_checks` (`inference_api.rs`): derives `weights_materialized` from `resource_bindings.weights` non-emptiness, `provider_ready` from the pinned Provider (if any) resolving with an execution API, `device_ready` from the pinned Device (if any) being found and `Available`. Each ANDed with the caller's own value (caller may force stricter, never looser).
- [x] 2.3 `warm_model_instance` calls `derive_effective_readiness_checks` before delegating to `ModelInstance::warmup`.
- [x] 2.4 `kernel_preparation_ready`/`autotuning_ready`/`adapter_ready`/`memory_pressure`/`runtime_policy_allows`/`residency_available`/`browser_supported` deliberately left caller-supplied -- no generic Runtime-side derivation exists for these today; documented in design.md's Non-Goals rather than fabricated.

## 3. Lifecycle/readiness consistency (item 2 + the `WarmupPolicy::Disabled` root cause)

- [x] 3.1 `ModelInstance::validate_readiness` no longer allows a computed `Ready` readiness to be published unless the lifecycle already supports it (`supports_inference_use()`) or is `Warming` (the in-flight state the normal warmup path is in when it calls this method, before transitioning to `Ready` itself).

## 4. Lifecycle+readiness safety net (item 4)

- [x] 4.1 `ModelInstance::acquire_usage` requires both `lifecycle.supports_inference_use()` and `readiness.accepts_generation()`.
- [x] 4.2 `ModelInstanceManager::generation_reference` requires the same.

## 5. Direct `mark_ready` bypass (item 5)

- [x] 5.1 Investigated closing via Rust visibility (`pub(crate)`): would require touching 13 integration-test call sites in a separate compilation unit (`magnetar-runtime/tests/contract_tests/model_instance.rs`) plus 2 in `tests.rs`, for a partial close (`mark_ready` has no Runtime context to derive `provider_ready`/`device_ready` even if hardened). Confirmed `magnetar-cli` never calls it directly (no breakage there). Decision, confirmed with the auditor: **not closed this way for this PR** -- documented as an accepted, explicit trade-off in design.md, not a silent gap. Item 4's lifecycle+readiness gate is the mitigation that applies regardless of this path.

## 6. Tests (item 6)

- [x] 6.1 `inference_api_warm_model_instance_rejects_forged_weights_materialized_claim` -- audit tests 18.1 + 18.3 (no materialization, default caller checks, must not reach Ready).
- [x] 6.2 `inference_api_warm_model_instance_disabled_policy_cannot_forge_ready_readiness` -- audit test 18.2, isolated from the weights check by directly binding a fake weight resource so only the lifecycle-consistency gate is under test.
- [x] 6.3 `model_instance_acquire_usage_rejects_ready_readiness_with_incompatible_lifecycle` -- audit test 18.4, forcing the inconsistent state directly (bypassing every public entry point) to prove the gate itself, independent of how the inconsistency might arise.
- [x] 6.4 `inference_api_warm_model_instance_reaches_ready_when_weights_and_provider_are_real` -- audit test 18.5, the happy path (real weights bound, a registered Reference CPU Provider pinned) still reaches Ready and accepts usage.
- [x] 6.5 Fixed one pre-existing test (`inference_api_model_instance_warmup_reports_lifecycle_conflict_when_already_ready`) that would otherwise have kept passing for the wrong reason (both the original already-Ready conflict and the new weights-derivation failure map to the same broad `ModelInstanceUnavailable` error variant) -- bound a real weight resource first so the test isolates its own stated concern again.

## 7. Verification (item 7)

- [x] 7.1 `cargo build -p magnetar-runtime --lib` clean.
- [x] 7.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`, `cargo fmt --all --check`, `CARGO_INCREMENTAL=0`. Both clean.
- [x] 7.3 `cargo test --locked --workspace --all-targets --all-features`: 64 + 1151 + 182 passed, 0 failed.
- [x] 7.4 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime`. Clean.
- [x] 7.5 Coverage ratchet: 78.91%, above the 78.89% baseline (the 4 new tests exercise real new production branches).
- [x] 7.6 `openspec validate --all --strict`. 78/78 passed.
- [ ] 7.7 Push, confirm a full green CI run via `gh run view --json status,conclusion` directly (never a piped `gh run watch`).
- [x] 7.8 Live `magnetar run qwen-test "Hello"` unaffected (production path never calls `warm_model_instance`, confirmed by inspection before this Change started, and reconfirmed by the live run producing the same output).

## 8. Close out

- [ ] 8.1 Archive this Change.
- [ ] 8.2 Update `CHANGELOG.md`'s Architecture Freeze #1 note per the auditor's own criterion (the freeze reverts to candidate until this fix + green CI, then accepted again).
- [ ] 8.3 File issues for the two explicitly-deferred items: the extended `#41`-style issue for `release_tensor()`'s missing structured error channel (the auditor's stated preference: fold into `#41` rather than a new issue), and confirm whether `model-loading-materializes-weight-resources` is genuinely superseded (the auditor flagged this as independent housekeeping, not blocking).
