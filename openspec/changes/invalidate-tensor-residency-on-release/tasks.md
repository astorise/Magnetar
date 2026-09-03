## 1. Spec

- [x] 1.1 Strengthen `memory`'s "Memory Manager Releases Model Residency" requirement so releasing a resource's residency explicitly means `tensor_residency()` returns `None` for it afterward, not only that the underlying `MemoryAllocation`'s state changes. `openspec validate --strict invalidate-tensor-residency-on-release` passes.

## 2. `MemoryManager::remove_tensor_residency`

- [x] 2.1 Add `remove_tensor_residency(&mut self, tensor: &TensorResourceId) -> Option<TensorResidency>` next to `record_tensor_residency`/`tensor_residency`, mirroring the setter's minimal style (plain `BTreeMap::remove`, no observation event).

## 3. Wire into rollback and unload

- [x] 3.1 `WeightMaterializationTransaction::abort` (`first_native_runtime.rs`): remove each staged weight's residency record, after releasing its Provider tensor and before releasing its Memory Manager allocation.
- [x] 3.2 `Runtime::unload_model_instance` (`runtime.rs`): remove each released weight resource's residency record, after resolving its owning Provider through that same record and releasing its Provider tensor (order matters: the Provider lookup reads `tensor_residency()`, so removal must come after that read), and before releasing its Memory Manager allocation.

## 4. Tests

- [x] 4.1 Rollback: `check_weight_materialization_failure_never_reaches_ready` (`first_native_runtime.rs`) gains a `tensor_residency(id).is_none()` assertion alongside its existing Provider-storage-is-gone assertion.
- [x] 4.2 Unload: `check_unload_releases_weight_resource_allocations` gains the same assertion.
- [x] 4.3 Repeated load/unload: `check_repeated_load_unload_does_not_accumulate_weight_storage` gains the same assertion per cycle, proving residency metadata does not accumulate the same way Provider storage was already proven not to.

## 5. Verification

- [x] 5.1 `cargo build -p magnetar-runtime --lib` clean.
- [x] 5.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`, `cargo fmt --all --check` (workspace), both with `CARGO_INCREMENTAL=0` given this session's own prior miss from stale incremental artifacts. Both clean.
- [x] 5.3 `cargo test --locked --workspace --all-targets --all-features` (all three binaries: magnetar-cli, magnetar-runtime lib, `contract_tests` integration). 64 + 1147 + 182 passed, 0 failed.
- [x] 5.4 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime`. Clean.
- [x] 5.5 Coverage ratchet: initially **failed** (78.88% vs 78.89% baseline) after adding the three `tensor_residency(id).is_none()` assertions -- their error arms are new lines that only execute if the fix is broken, so they don't execute in a passing run, and `magnetar-runtime/src/tests.rs` (where a first attempt to offset this with a direct `remove_tensor_residency` unit test was added) turned out to be excluded from this measurement entirely by cargo-llvm-cov's own built-in `(tests\.rs|...[_-]tests\.rs)$` pattern (documented in `quality/coverage-baseline.json`'s `exclusions`), so that unit test contributed nothing to the number. Fixed properly, not by adjusting the baseline: consolidated the three duplicated assertion blocks into one shared `assert_tensor_residency_absent` helper (real deduplication, not a coverage-chasing contortion) -- fewer total new uncovered lines restored the ratchet to exactly 78.89%, matching baseline.
- [x] 5.6 `openspec validate --all --strict`. 78/78 passed.
- [ ] 5.7 Push, confirm a full green CI run without relying on a piped `gh run watch` exit code (this session's own two prior masked-failure incidents on this exact branch).
- [ ] 5.8 Live `magnetar run qwen-test "Hello"` still works.

## 6. Close out

- [ ] 6.1 Archive this change.
- [ ] 6.2 Update `CHANGELOG.md`'s Architecture Freeze #1 note to record this fix and the fresh CI run, per this revalidation audit's own criterion (`Architecture Freeze #1 = ACCEPTED` only "après correction du cleanup TensorResidency et CI verte").
