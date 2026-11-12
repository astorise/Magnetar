## 1. Inventory and baseline

- [ ] 1.1 Generate the module dependency graph of `magnetar-runtime` (non-test code) and list every cycle with the items that cause it.
- [ ] 1.2 Write `quality/layers.toml`: each module, its target layer, and for modules that cannot reach their target, the blocking dependency. Settle where `compute` and `provider` go.
- [ ] 1.3 Record the baseline: public API listing of `magnetar-runtime`, test names and counts per suite, coverage, clean and incremental build times.

## 2. Break dependency cycles (inside `magnetar-runtime`)

- [ ] 2.1 Roadmap modules: move the types used by `kernel_artifact`, `conformance` and `production_model_ingestion` to their owning modules; re-export them from `provider_roadmap` and `model_format_roadmap`.
- [ ] 2.2 Core: remove `tensor` → `compute` and `affinity` → `compute`/`provider`.
- [ ] 2.3 Graph and memory: remove `scheduler` ↔ `runtime` ↔ `planning` and the `planning`/`scheduler` references to `compute` and `provider`.
- [ ] 2.4 Kernel pipeline: remove `kernel_registry`, `kernel_selection_policy`, `kernel_autotuning` → `model_instance`/`session`, and `kernel_optimization_orchestration` → `inference_api`/`cli_boundary`.
- [ ] 2.5 Component engine: remove `component` → `model` and `graph_builder_capability` → `compute`.
- [ ] 2.6 Orchestration: remove `provider` → `inference_api` and `model_loading` → `first_native_runtime`.
- [ ] 2.7 After each task above: full test suites pass, public API listing shows no removal.

## 3. Layering gate

- [ ] 3.1 Add the layering check tool to the root workspace, with unit tests for upward dependency, unmapped module and allowed same-layer dependency.
- [ ] 3.2 Run it in `quality.yml` on every pull request and document the local command in `CONTRIBUTING.md`.

## 4. Split `first_native_runtime` and monolithic tests

- [ ] 4.1 Confirm no open pull request touches `first_native_runtime.rs`; otherwise wait for it to merge.
- [ ] 4.2 Convert `first_native_runtime` to a directory module and move code into the submodules of design D6, one move-only commit per submodule.
- [ ] 4.3 Move each test of `first_native_runtime/tests.rs` next to the code it tests; widen visibility to `pub(super)`/`pub(crate)` only.
- [ ] 4.4 Split `src/tests.rs`: tests using only public items go to `tests/contract_tests/`, the others to their module.
- [ ] 4.5 Verify: no first-native submodule above 2,500 non-test lines; same test names and counts as the baseline; update the file lists and comments in `quality.yml` and `quality/coverage-baseline.json`.

## 5. Extract layer crates

- [ ] 5.1 `magnetar-core`: create the crate, move layer 1 modules, re-export from `magnetar-runtime` at the same paths.
- [ ] 5.2 `magnetar-graph`: same for the graph and memory modules.
- [ ] 5.3 `magnetar-kernel`: same for the kernel pipeline modules.
- [ ] 5.4 `magnetar-component-engine`: same for the Component engine modules, `wit/` and the Component fixtures; forward the `wasmtime-component-engine` and `web-component-engine` features.
- [ ] 5.5 After each extraction: compare the public API listing with the baseline, and report clean and incremental build times.
- [ ] 5.6 Cross-platform: build each crate on Linux, macOS and Windows, and `magnetar-core`, `magnetar-graph`, `magnetar-kernel` and the web Component engine for `wasm32`.

## 6. Consumers and validation

- [ ] 6.1 Build and test `magnetar-cli`, `inference-components`, `roadmap-contracts` and the three integration test crates without source change.
- [ ] 6.2 Build and test each pinned Component, Provider, Format and Loader repository against the result without source change.
- [ ] 6.3 Run the WIT validation gate and compare the WIT packages with the baseline (identical names, interfaces, versions).
- [ ] 6.4 Run GPU CI (`cuda-first-native`, `multi-device-cpu-cuda`, real-checkpoint numerics): results identical to the baseline.

## 7. Release, coverage and documentation

- [ ] 7.1 `tools/release-publish` and `release-publication.yml`: package and publish the five crates in dependency order with one version; dry run covers all five.
- [ ] 7.2 `tools/coverage-ratchet` and `quality/coverage-baseline.json`: measure all Runtime crates; covered lines not lower than the baseline.
- [ ] 7.3 Extend formatting, Clippy, test, documentation and `cargo-deny` gates to the new crates.
- [ ] 7.4 Update `README.md`, `CONTRIBUTING.md`, the architecture documents under `docs/` and `openspec/config.yaml` with the crate layout and the layer map.
- [ ] 7.5 Add a `CHANGELOG.md` entry stating that public paths are unchanged and listing the new crates.
