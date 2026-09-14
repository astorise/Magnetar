## 1. Fixture

- [x] 1.1 Scaffolded a synthetic Component crate (outside `components/qwen`, in the session scratchpad) implementing the real `model-component-graph-producer` world with a deliberately degenerate graph (`embedding -> rmsnorm -> matmul`, no decoder layers).
- [x] 1.2 Built it through the real toolchain: `cargo build --target wasm32-unknown-unknown --release`, then `wasm-tools component new` to produce a real WASM Component.
- [x] 1.3 Validated it (`wasm-tools validate`) and confirmed its WIT world is structurally identical to the real Qwen Component's (`wasm-tools component wit` on both, same imports/exports).
- [x] 1.4 Copied it into `magnetar-runtime/fixtures/components/synthetic-minimal.component.wasm` with a manifest (`synthetic-minimal.component.wasm.magnetar-component.yaml`) mirroring `qwen-real.component.wasm`'s own manifest format, digest computed from the real checked-in bytes.

## 2. Implementation

- [x] 2.1 `magnetar-runtime/src/first_native_runtime.rs`: added `SYNTHETIC_MINIMAL_COMPONENT_BYTES`/`SYNTHETIC_MINIMAL_COMPONENT_MANIFEST_BYTES` `#[cfg(test)]` constants via `include_bytes!`, mirroring the existing `QWEN_REAL_COMPONENT_BYTES` pattern. No production code path changes.

## 3. Tests

- [x] 3.1 Extended the existing consolidated registry test (`register_inference_component_artifact_enforces_trust_is_idempotent_and_matches_the_singleton_path`) to also register the synthetic Component and assert its graphs differ from Qwen's (node counts, operator-sequence hashes) and that building them does not disturb Qwen's own continued correctness.
- [x] 3.2 A first attempt used a separate `#[test]` function; reproduced a real digest-sharing race across parallel tests locally (2 of 4 full-suite runs failed) before folding the assertions into the existing consolidated test instead, per its own documented reasoning.
- [x] 3.3 `cargo test -p magnetar-runtime --features wasmtime-component-engine --lib` run 4 consecutive times, all green (1261 tests).
- [x] 3.4 `cargo fmt`, `cargo clippy -p magnetar-runtime --features wasmtime-component-engine --tests -- -D warnings`, `cargo check -p magnetar-runtime --target wasm32-unknown-unknown --all-features` all clean.

## 4. Documentation

- [x] 4.1 `openspec validate add-second-component-fixture-for-registry-multiplicity-proof --strict` passes.
- [x] 4.2 README/`docs/audits/audit-magnetar-integration-tachyon-2026-09-13.md` updated once archived: MAG-07 moves from open to closed.
