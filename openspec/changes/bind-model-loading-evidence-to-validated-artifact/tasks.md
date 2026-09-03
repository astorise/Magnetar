## 1. Spec

- [x] 1.1 `model-loading`: new "Loaded Model Context Is Runtime-Issued" requirement.
- [x] 1.2 `model-instance`: "Model Instance Readiness" strengthened for Runtime-issued materialization evidence; new "Model Instance Weight Resource Bindings Are Runtime-Sealed" requirement. `openspec validate --strict bind-model-loading-evidence-to-validated-artifact` passes.

## 2. Materialization evidence storage

- [ ] 2.1 Add a `MaterializationEvidence` record type (artifact id + committed resource-id set) and Runtime-owned storage keyed by `ModelInstanceId`, empty/unpopulated for now -- additive only, no visibility changes yet.
- [ ] 2.2 `cargo build -p magnetar-runtime --lib` clean before proceeding.

## 3. Seal `LoadedModelContext` (audit P0-A, loading half)

- [ ] 3.1 `LoadedModelContext`/`ModelLoadingResidencyPlan` fields become `pub(crate)`.
- [ ] 3.2 `cargo build --workspace --all-targets --all-features` clean. Expected: no call sites outside `model_loading.rs` need changes (confirmed by grep during design -- `contract_tests/model_instance.rs`'s `loaded_context()` already calls `ModelLoadingCoordinator::load()`, not a struct literal). If a build error surfaces anywhere else, treat it as new information, not a design failure -- fix it, and note in this Change's proposal/design why the design's grep missed it.

## 4. Public materialization entrypoint + evidence minting (audit P0-A, provenance half)

- [ ] 4.1 Promote `materialize_model_instance_weights` (or a thin `pub` wrapper around it) to a public `Runtime` method with the same `(instance: &ModelInstanceId, artifact_owner: &str, weights: &BTreeMap<String, HostTensor>) -> Result<(), InferenceApiError>` shape.
- [ ] 4.2 `WeightMaterializationTransaction::commit` mints/replaces the instance's `MaterializationEvidence` (this instance's `definition.artifact` + the exact committed resource-id set) as part of its existing binding-write step.
- [ ] 4.3 `WeightMaterializationTransaction::abort` and `Runtime::unload_model_instance` clear the instance's evidence record, mirroring existing `TensorResidency` cleanup (`invalidate-tensor-residency-on-release`).

## 5. Seal weight resource bindings (audit P0-A, provenance half)

- [ ] 5.1 `ModelInstanceDefinition.resource_bindings` and all four `ModelInstanceResourceBindings` fields (`weights`, `memory_allocations`, `released_memory_allocations`, `released_provider_resources`) become `pub(crate)`.
- [ ] 5.2 `contract_tests/model_instance.rs`'s `bind_fake_weight` migrated onto the step-4 public entrypoint instead of direct field mutation; confirm it still exercises a real `ReferenceCpuProvider` write (no behavior regression versus today's already-real `write_tensor` call, just routed through the authorized transaction instead of hand-assembled around it).
- [ ] 5.3 `cargo build --workspace --all-targets --all-features` clean; fix any other external call sites the design's grep did not find.
- [ ] 5.4 If a real external read dependency on `resource_bindings`'s sealed fields surfaces (not expected per the design's grep, which found only writes), add the narrow read-only accessor design.md's Risks section anticipates (e.g. `ModelInstanceDefinition::bound_weight(name) -> Option<&TensorResourceId>`) rather than reopening field visibility.

## 6. Readiness derivation switch

- [ ] 6.1 `derive_effective_readiness_checks`'s `weights_materialized`: replace the `ProviderExecutionApi::read_tensor` probe with a check that materialization evidence exists for this instance, its recorded artifact id matches `definition.artifact`, and its recorded resource-id set matches the current `resource_bindings.weights` values exactly.
- [ ] 6.2 Confirm no remaining production or test code path depends on `read_tensor` for readiness purposes (it may still be used elsewhere, e.g. actual tensor computation -- only its readiness-proof usage is removed).

## 7. Tests

- [ ] 7.1 Forged-context rejection: confirm (by the type system, not a runtime test -- `LoadedModelContext`'s sealed fields make this a compile-time impossibility for external code) that `contract_tests` cannot construct one directly; add/keep a `tests.rs` unit test proving `create_model_instance` only ever sees Runtime-produced contexts in this crate's own suite too.
- [ ] 7.2 Hand-written weight binding rejected: a caller performs a real `write_tensor` + real `record_tensor_residency`, then attempts to bind a weight through whatever internal-only path remains reachable in `tests.rs` (same-crate access) without going through the authorized transaction -- must not reach Ready, and readiness must report the specific "no matching evidence" reason.
- [ ] 7.3 Cross-instance evidence reuse rejected: instance A materializes for real; instance B's bindings are set to match A's (same-crate test, since this needs pre-sealing-equivalent access); B must not become Ready.
- [ ] 7.4 Cross-artifact evidence mismatch rejected: an instance's evidence was minted while it declared a different `ModelArtifactId` than it currently does -- must not count as materialized. (Note per design.md: not reachable via any real code path today since `artifact` has no setter; implement as a defense-in-depth unit test against the check itself, not a claimed real-world bypass.)
- [ ] 7.5 Device-only-Provider happy path: a test `ProviderExecutionApi` whose `read_tensor` always returns `None` (simulating no host readback) still reaches `weights_materialized: true` through the authorized transaction, proving the P1 (`read_tensor`/`HostTensor` dependency) is closed.
- [ ] 7.6 Existing test suite audit: every test currently relying on `read_tensor`-based readiness proof (round 5's tests) updated to go through the new evidence path instead; confirm none of them silently pass for the wrong reason.
- [ ] 7.7 Live production path (`bind_qwen_fixture_weights` -> `materialize_model_instance_weights` -> commit) still reaches Ready exactly as before -- no observable behavior change for the one real production caller.

## 8. Verification

- [ ] 8.1 `cargo build --locked --workspace --all-targets --all-features` clean.
- [ ] 8.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` clean.
- [ ] 8.3 `cargo fmt --all -- --check` clean.
- [ ] 8.4 `cargo test --locked --workspace --all-targets --all-features`: full suite passing, count recorded here.
- [ ] 8.5 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean.
- [ ] 8.6 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean.
- [ ] 8.7 Coverage ratchet: at or above baseline (no lowering the baseline to pass).
- [ ] 8.8 `openspec validate --all --strict` passing, with the exact total item count recorded here so a future archive-merge regression (like `9939232`'s) is easier to notice.
- [ ] 8.9 Live `magnetar run qwen-test "Hello"` unaffected.
- [ ] 8.10 Push, confirm a full green CI run via `gh run view --json status,conclusion` and the jobs list directly (never a piped `gh run watch`).

## 9. Close out

- [ ] 9.1 Diff the canonical spec files this Change touches before and after archiving (matching the exact check this Change's own P0-B discovery used) to confirm the archive-merge did not silently drop any requirement or scenario.
- [ ] 9.2 Archive this Change.
- [ ] 9.3 Update `CHANGELOG.md`'s Architecture Freeze #1 note: this Change closing means P0-A is resolved; note explicitly that byte-content provenance (this Change's Non-Goal) remains open and name the proposed follow-up.
- [ ] 9.4 Consider whether to open the follow-up Change (manifest-level per-tensor/aggregate content digest verification) now or defer -- a decision for the user, not made here.
