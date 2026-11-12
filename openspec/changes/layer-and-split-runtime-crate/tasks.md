## 1. Inventory and baseline

- [x] 1.1 Generate the module dependency graph of `magnetar-runtime` (non-test code) and list every upward dependency with the items that cause it (`layering-check --report`, `--cycles`).
- [x] 1.2 Write `quality/layers.toml`: each module's layer, the modules held in Runtime orchestration with the reason, and the debt list. `compute` and `provider` stay in Runtime orchestration.
- [x] 1.3 Record the test baseline: 1,178 unit tests and 177 contract tests in `magnetar-runtime`, all passing.
- [x] 1.4 Public API baseline: the commit before the first move is `d5231c7`; `cargo public-api -p magnetar-runtime -sss diff d5231c7..HEAD` lists removed, changed and added items.
- [x] 1.5 Build times at `d5231c7`, one run on a 4-core container: clean `cargo check -p magnetar-runtime` 53.1 s; after touching one Runtime file 3.0 s; clean test build 88.5 s.

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
- [x] 4.7 The test build without `wasmtime-component-engine` compiles: `first_native_runtime` imports its graph oracle from `tests` under the same cfg. Seven tests fail in that configuration; they could not be built before, and no CI job runs it.

## 5. Extract layer crates

- [x] 5.1 `magnetar-core`: the ten core modules; `magnetar-runtime` re-exports each module at its former path.
- [x] 5.2 `magnetar-graph`: `execution_graph`, `graph_fusion`, `memory`. Memory Manager inspection helpers used by Runtime tests sit behind a `test-support` feature.
- [x] 5.3 `magnetar-model`: `model`, `model_identity`. `ModelTrustDecision` keeps its crate-private fields and constructors (design D9).
- [x] 5.4 `magnetar-kernel`: the kernel pipeline modules, `execution_stream` and the Kernel Bundle fixtures.
- [x] 5.5 `magnetar-engine`: the Component engine modules, with the `wasmtime-component-engine` and `web-component-engine` features forwarded from `magnetar-runtime`. `wit/` and the Component fixtures stay under `magnetar-runtime` (design D10).
- [x] 5.6 Tests that need a higher layer stay in `magnetar-runtime`: graph fusion against the real Qwen Component, Kernel Registry against the Reference CPU Provider, one Component engine capability check.
- [x] 5.7 After the extractions: every one of the 4,327 public type, trait, function, constant and module paths of the baseline (`d5231c7`) still resolves through `magnetar-runtime` (compiled probe); 1,703 workspace tests pass, none removed; Clippy, documentation and layering gates pass.
- [x] 5.8 `wasm32-unknown-unknown`: `cargo check -p magnetar-runtime --all-features` passes with the layer crates.
- [x] 5.9 Build times after the split, same machine, one run: clean check 44.6 s (53.1 s before); after touching a Runtime file 1.3 s, a `magnetar-kernel` file 2.0 s, a `magnetar-core` file 2.3 s (3.0 s before for any file); clean test build 72.4 s (88.5 s before).
- [ ] 5.10 Build each crate on macOS and Windows (CI matrix).

## 6. Consumers and validation

- [x] 6.1 `magnetar-cli`, `inference-components`, `roadmap-contracts` and the three integration test crates build without source change; `magnetar-cli` (93) and `magnetar-inference-component` (14) tests pass. `submodule-workspace/Cargo.lock` is updated.
- [x] 6.2 Pinned Provider, Format and Loader repositories: each `Cargo.lock` gains the five layer crates (one pull request per repository, no source change), and the eleven submodule pins point at those commits. With them, `cargo test --locked` passes in each of the eleven. Once the pull requests merge, re-pin to the merged commits. The Component repositories do not depend on `magnetar-runtime`.
- [ ] 6.3 Run the WIT validation gate in CI. The WIT files are not moved or edited by this change.
- [ ] 6.4 Run GPU CI (`cuda-first-native`, `multi-device-cpu-cuda`, real-checkpoint numerics): results identical to the baseline.

## 7. Release, coverage and documentation

- [x] 7.1 `release-publication.yml` packages and publishes the six crates together; `cargo publish --dry-run` of the six passes locally, in dependency order.
- [ ] 7.2 Coverage: confirm in CI that the workspace measurement is not below `quality/coverage-baseline.json` now that code sits in six crates. (The SBOM is generated from `cargo metadata` and lists the layer crates without change.)
- [x] 7.3 Formatting, Clippy, test, documentation and `cargo-deny` gates run on the workspace and so cover the new crates; the layering, model-family isolation, test-placement and Wasmtime jobs name them explicitly.
- [x] 7.4 `README.md`, `CONTRIBUTING.md`, `docs/quality.md`, `openspec/config.yaml` and `quality/layers.toml` describe the crate layout.
- [x] 7.5 `CHANGELOG.md` entry: public paths unchanged, new crates listed.
