## 1. Split the generation engine inside `magnetar-runtime`

- [ ] 1.1 Record the baseline: Runtime unit and contract test counts, `providers/cuda` and integration-test build success, clean and incremental build times of `magnetar-runtime` and `providers/cuda`.
- [ ] 1.2 Move `first_native_runtime.rs` to `first_native_runtime/mod.rs` and split it into `fixture`, `component_graphs`, `graph_cache`, `dispatch`, `execution`, `step_replay`, `weights`, `placement` and `production`, re-exporting every public item at its current path.
- [ ] 1.3 Split `first_native_runtime/tests.rs` into per-module `tests.rs` files; move shared helpers to a test-only `test_support` module.
- [ ] 1.4 Record a size budget per engine module in the design and a check (script or test) that reports modules over budget.
- [ ] 1.5 Verify: same test counts and outcomes, `cargo fmt`/`clippy -D warnings`, the generic facade isolation guard, every submodule and integration crate compiling unchanged, and one GPU workflow run with unchanged token checks.

## 2. Extract crates

- [ ] 2.1 Measure the Provider closure: every `magnetar_runtime` item imported by `providers/*`, plus every item their public signatures require; record it in the design.
- [ ] 2.2 Create `magnetar-provider-api` with that closure (including the Provider-facing operator semantics from `reference_cpu`); make `magnetar-runtime` depend on it and re-export each moved module at its old path.
- [ ] 2.3 Create `magnetar-kernel` with the kernel artifact, ingestion, signature, compilation, benchmark, performance model, autotuning, qualification, cache, selection and orchestration modules; re-export from `magnetar-runtime`.
- [ ] 2.4 Create `magnetar-component-host` with the Wasmtime and web Component hosts and Capability linking; re-export from `magnetar-runtime`; keep the `wasmtime-component-engine` and `web-component-engine` features working on native and `wasm32` targets.
- [ ] 2.5 Add a dependency-direction check (for example `cargo metadata` in CI) asserting no extracted crate depends on `magnetar-runtime` and the API crate depends on no other Magnetar crate.
- [ ] 2.6 Verify after each extraction: full Runtime tests, every Provider, Loader, Format and Component submodule, `inference-components`, `magnetar-cli` and the integration tests compiling unchanged; WIT validation unchanged; build times compared with 1.1; Linux, Windows and macOS CI and the `wasm32` build passing.

## 3. Roadmap modules

- [ ] 3.1 Move the live metadata types and validators of `model_format_roadmap` to `model_format`, re-exported at the old path; update in-repo users to the new path.
- [ ] 3.2 Move the roadmap contract items of `model_format_roadmap` and all of `provider_roadmap` to `roadmap-contracts` with their tests; update the doc references in `conformance.rs`, `kernel_artifact.rs` and `model_format_roadmap.rs`.
- [ ] 3.3 Verify `loaders/huggingface` compiles unchanged and the moved tests pass in `roadmap-contracts`.

## 4. Documentation

- [ ] 4.1 Update the architecture documentation and `SUBMODULES.md` with the crate DAG and the rule that Providers depend on the provider API, never the reverse.
- [ ] 4.2 Record in this change's design which paths are re-exports scheduled for removal, for the follow-up change that removes them.
