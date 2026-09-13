## 1. Investigation

- [x] 1.1 Traced the real call chain from `run_first_native_generation` down to confirm the WASM Component genuinely is invoked via `WasmtimeComponentEngine` in production builds (not a parallel/unused path) -- `first_native_component_graphs_for_prompt` -> `build_first_native_graphs_from_real_qwen_component` -> `qwen_real_component_runtime`.
- [x] 1.2 Confirmed the real root cause: `register_qwen_component_artifact` always stamps pushed bytes with the hardcoded `QWEN_REAL_COMPONENT_DIGEST` constant, and `qwen_real_component_runtime` is a single process-wide `OnceLock` -- a caller-supplied Component with genuinely different bytes fails digest verification the first time generation reaches it, not at registration time.
- [x] 1.3 Confirmed `build_first_native_graphs_for_config`'s body is already fully parameterized by `config`/`identity`, never specific to which compiled Component binary drives it -- the fix is in *which runtime* gets used, not in graph-production logic itself.

## 2. Implementation

- [x] 2.1 `magnetar-runtime/src/first_native_runtime.rs`: `RegisteredComponentRuntime` struct, `REGISTERED_COMPONENT_RUNTIMES` keyed registry (`Mutex<BTreeMap<String, Arc<RegisteredComponentRuntime>>>`).
- [x] 2.2 `register_inference_component_artifact(component_bytes, manifest_bytes, trust: &ComponentTrustStore) -> Result<ComponentDigest, E2eConformanceError>`: real digest computation, caller-supplied trust, idempotent per digest.
- [x] 2.3 `named_component_runtime(digest)` and `build_first_native_graphs_from_named_component(digest, config, identity, prompt_token_count)`.
- [x] 2.4 Extracted `build_first_native_graphs_for_config`'s body into `build_first_native_graphs_with_runtime`, called by both the pre-existing singleton path (unchanged behavior) and the new generic path.

## 3. Tests

- [x] 3.1 `register_inference_component_artifact_computes_real_digest_and_enforces_caller_trust`: an empty trust store rejects; a trust store naming the artifact's own real digest accepts; the returned digest matches an independently computed `ComponentDigest::sha256`.
- [x] 3.2 `register_inference_component_artifact_is_idempotent_per_digest`.
- [x] 3.3 `build_first_native_graphs_from_named_component_matches_the_real_qwen_component_path`: the load-bearing correctness proof -- graphs built via the new generic path have identical operator-sequence hashes (not just node counts) to graphs built via the pre-existing hardcoded singleton path, for the identical real `qwen-real.component.wasm` fixture bytes. Passed on the first run.
- [x] 3.4 `build_first_native_graphs_from_named_component_fails_closed_for_an_unregistered_digest`.
- [x] 3.5 Full regression: `cargo test -p magnetar-runtime --lib` (1262 passed, up from 1258, zero regressions), `cargo clippy -p magnetar-runtime --all-targets -- -D warnings` clean, `cargo fmt --check` clean.
- [x] 3.6 Fixed an unrelated compile break surfaced by reconciling `codex/tachyon-component-boundary` with `main`: `integration-tests/production-loading`'s 14 `ProductionGenerationRequest` struct literals were missing the `max_generation_millis` field that branch added; added `max_generation_millis: None` to each. Verified the crate builds clean.

## 4. Documentation

- [x] 4.1 `openspec validate wire-generic-inference-component-runtime --strict` passes.
- [x] 4.2 `design.md` documents the deferred follow-up phase (wiring `inference-components` to this registry, closing the audit's MAG-01/02/03 for real) and the untested-end-to-end two-real-Components gap, honestly.
