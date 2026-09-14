## 1. Investigation

- [x] 1.1 Confirmed `ProductionQwenLoadedModel::generate`/`generate_streaming` both take `&mut self` -- the concurrency model MAG-06 asked to be "validated" was already settled and compiler-enforced, not an open design question this change needed to decide from scratch.

## 2. Implementation

- [x] 2.1 `inference-components/src/lib.rs`: `LoadedInferenceComponent` gains a "Concurrency model" doc comment stating the real, pre-existing behavior (one generation in flight per instance, a second call blocks rather than being rejected/interleaved, concurrent generation across requests needs multiple loaded instances). `invoke_payload`/`invoke_payload_streaming` each gain a one-line pointer to it. Zero non-comment lines changed.

## 3. Tests

- [x] 3.1 `cargo build`/`cargo clippy --all-targets -- -D warnings`/`cargo fmt --check`/`cargo doc --no-deps` all clean (the intra-doc link to `LoadedInferenceComponent` resolves). Pre-existing 3 tests pass unmodified (no behavior to newly test -- this change adds no new behavior).

## 4. Documentation

- [x] 4.1 `openspec validate document-inference-component-concurrency-model --strict` passes.
- [x] 4.2 README/`docs/audits/audit-magnetar-integration-tachyon-2026-09-13.md` updated once archived: MAG-06 moves from open to closed.
