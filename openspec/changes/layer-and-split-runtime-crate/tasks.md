## 1. Inventory and baseline

- [x] 1.1 Generate the module dependency graph of `magnetar-runtime` (non-test code) and list every upward dependency with the items that cause it (`layering-check --report`, `--cycles`).
- [x] 1.2 Write `quality/layers.toml`: each module's layer, the modules held in Runtime orchestration with the reason, and the debt list. `compute` and `provider` stay in Runtime orchestration.
- [x] 1.3 Record the test baseline: 1,178 unit tests and 177 contract tests in `magnetar-runtime`, all passing.
- [x] 1.4 Public API baseline: the commit before the first move is `d5231c7`; `cargo public-api -p magnetar-runtime -sss diff d5231c7..HEAD` lists removed, changed and added items.
- [ ] 1.5 Record clean and incremental build times before the first crate extraction.

## 2. Layering gate

- [x] 2.1 Add `tools/layering-check` to the root workspace, with unit tests for upward dependency, same-rank dependency, unmapped module, test-only code, re-exports, glob imports, macros and the debt list.
- [x] 2.2 Run it in `quality.yml` on every pull request and document the local command in `CONTRIBUTING.md`.

## 3. Remove upward dependencies (inside `magnetar-runtime`)

- [x] 3.1 `compute`: move the dtype, layout, shape, view and tensor descriptors, the compute identifiers, `ComputeValidationError` and `redact_backend_diagnostic` to the core module `compute_types`; re-export from `compute`.
- [x] 3.2 `affinity`: move the bindings, `ResourceAffinity`, `AffinityConstraints` and `AffinityError` to `binding`, and the health, lifecycle, pressure and admission types to `provider_status`; re-export from `affinity`.
- [x] 3.3 `planning`: move `MemoryPlanningError` and `MemoryPressureReport` to `compute_types`; re-export from `planning`.
- [x] 3.4 `tensor`, `kernel`, `execution_stream`: move `TensorError`, the tensor intent and memory classes, `KernelMemoryClass` and `CompletionTokenId` to `compute_types`; re-export from their old modules.
- [x] 3.5 `observability`: move `CorrelationId`, `TraceId`, `SpanId` to `trace_id`; re-export.
- [x] 3.6 `scheduler` and `provider`: move the Provider execution error types to `provider_status` and the Provider ABI version and memory rule to `provider_abi`; re-export.
- [x] 3.7 `component`: move `WitInterface` to `capability`; re-export.
- [x] 3.8 After each task above: unit and contract suites pass with the baseline counts; Clippy clean with `-D warnings`.
- [x] 3.9a After tasks 3.1 to 3.7: public API diff against `d5231c7` shows no removed and no changed item; root and submodule workspaces build without source change; `magnetar-cli` (93) and `magnetar-inference-component` (14) tests pass.
- [x] 3.9 Model data (design D5b): add the `model-data` layer; assign `model` to it; move the Model Instance, session, adapter and tokenizer identities with their error enums to `model_identity`; move the kernel selection policy types to `kernel_selection_policy`; move `ComponentArtifactReference` to `binding`.
- [x] 3.10 `capability -> provider`: move `ProviderError`, `ScheduledOperationId` and the Provider ABI descriptor types to `provider_status` and `provider_abi`.
- [x] 3.11 Debt list empty; `[debt]` table removed from the layer map. Verified: gate clean, same 1,355 test names, public API diff against `d5231c7` with no removed and no changed item, root and submodule workspaces build, `magnetar-cli` and `magnetar-inference-component` tests pass.

## 4. Split `first_native_runtime` and monolithic tests

- [x] 4.1 Confirm no open pull request touches `first_native_runtime.rs` (astorise/Magnetar#167 merged; the two open pull requests are dependency updates).
- [x] 4.2 Convert `first_native_runtime` to a directory module with thirteen submodules (design D6). One scripted commit: items keep their order and bodies, and file-private items become `pub(super)`.
- [x] 4.3 Point the static guards that read the module's source text (seven tests, `first_native_implementation_cut`, `tools/check_generic_facade_family_isolation.py`) at the directory's production files.
- [x] 4.4 Verify: largest submodule 1,960 lines; the 1,355 test names are identical to the baseline; public API diff shows no removed and no changed item; Clippy, documentation and layering gates pass.
- [ ] 4.5 Move each test of `first_native_runtime/tests.rs` (11,771 lines) next to the code it tests.
- [ ] 4.6 Split `src/tests.rs`: tests using only public items go to `tests/contract_tests/`, the others to their module.
- [ ] 4.7 Fix the test build without the `wasmtime-component-engine` feature: `cargo check -p magnetar-runtime --no-default-features --all-targets` fails with six unresolved test-oracle references. It failed the same way before the split and no CI job runs it.

## 5. Extract layer crates

- [ ] 5.1 `magnetar-core`: create the crate, move the core modules, re-export from `magnetar-runtime` at the same paths.
- [ ] 5.2 `magnetar-graph`: same for the graph and memory modules.
- [ ] 5.2a Model data: extract `model` and `model_identity` as `magnetar-model`, or fold them into `magnetar-graph`'s dependents as decided from their size (design D5b).
- [ ] 5.3 `magnetar-kernel`: same for the kernel pipeline modules.
- [ ] 5.4 `magnetar-engine`: same for the Component engine modules, `wit/` and the Component fixtures; forward the `wasmtime-component-engine` and `web-component-engine` features.
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
- [ ] 7.4 Update `README.md`, the architecture documents under `docs/` and `openspec/config.yaml` with the crate layout and the layer map.
- [ ] 7.5 Add a `CHANGELOG.md` entry stating that public paths are unchanged and listing the new crates.
