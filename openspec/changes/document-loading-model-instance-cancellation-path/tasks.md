## 1. Spec

- [x] 1.1 `model-instance`: new "Loading Model Instance Can Be Canceled" requirement. `openspec validate --strict document-loading-model-instance-cancellation-path` passes.

## 2. Verify the existing capability

- [x] 2.1 Confirmed `ModelInstance::fail`/`invalidate` unconditionally set lifecycle regardless of current state (do not consult `allows_transition_to`).
- [x] 2.2 Confirmed `(Failed, Unloading)` is already a valid `allows_transition_to` pair, and `ModelInstanceManager::unload` already accepts `Failed`.
- [x] 2.3 New test `loading_instance_can_be_canceled_via_fail_then_unload` (`first_native_runtime/tests.rs`): creates an instance, never materializes it, fails it from `Loading`, unloads it, and asserts a clean report (`released_weight_resources`/`released_memory_allocations` both empty) and final `Unloaded` state.

## 3. Verification

- [x] 3.1 `cargo build --locked --workspace --all-targets --all-features` clean.
- [x] 3.2 `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` clean.
- [x] 3.3 `cargo fmt --all -- --check` clean.
- [x] 3.4 `cargo test --locked --workspace --all-targets --all-features`: 64+1186+173 passed, 0 failed.
- [x] 3.5 `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps` clean.
- [x] 3.6 `cargo check --target wasm32-unknown-unknown --all-features -p magnetar-runtime` clean (pre-existing unrelated warnings only).
- [x] 3.7 Coverage ratchet: 79.00% (baseline 78.89%).
- [x] 3.8 `openspec validate --all --strict`: 76/76.
- [x] 3.9 Live `magnetar run qwen-test "Hello"` unaffected: `[generated token ids: 239 239 239 239]`.
- [ ] 3.10 Push, confirm a full green CI run via `gh run view --json status,conclusion` and the jobs list directly (never a piped `gh run watch`).

## 4. Close out

- [ ] 4.1 Diff the canonical spec file this Change touches before and after archiving, to confirm the archive-merge did not silently drop anything.
- [ ] 4.2 Archive this Change.
- [ ] 4.3 Close GitHub issue "A Model Instance stuck in Loading cannot currently be unloaded or canceled" with a comment linking the archived Change and the new test.
