## Context

`magnetar-runtime` 1.0.0 is published on crates.io and consumed by 19 manifests: `magnetar-cli`, `inference-components`, `roadmap-contracts`, three integration test crates, and 13 external repositories pinned as submodules. They use both module paths (`magnetar_runtime::model::…`, `magnetar_runtime::provider::…`) and root re-exports (`magnetar_runtime::load_model`). `lib.rs` declares every module `pub` and re-exports almost all of them with `pub use module::*`.

The module graph has cycles that cross every layer the audit proposes (see the proposal). A crate boundary cannot contain a cycle, so cycles are removed first, inside the single crate, where each step compiles and is reviewable on its own.

## Goals / Non-Goals

**Goals:**

- A layered module graph with no upward dependency, enforced in CI.
- `first_native_runtime` split into modules with one responsibility each.
- Four lower-layer crates extracted, with `magnetar-runtime` as the public entry point.
- No source change required in any consumer.

**Non-Goals:**

- Behaviour change of any kind.
- A Provider crate, a CLI split, or new test coverage of CLI or resilience behaviour (see the proposal).

## Decisions

### D1. Layers

`quality/layers.toml` is the authoritative map. Summary:

| Layer (rank) | Crate | Modules |
|---|---|---|
| Core types (1) | `magnetar-core` | `capability`, `device`, `resolution`, `operator`, `artifact_signature`, and the modules this change creates to hold shared data types: `compute_types`, `binding`, `provider_status`, `provider_abi`, `trace_id` |
| Graph and memory (2) | `magnetar-graph` | `execution_graph`, `graph_fusion`, `memory` |
| Kernel pipeline (3) | `magnetar-kernel` | the `kernel*` modules except `kernel_optimization_orchestration`, plus `execution_stream` |
| Component engine (3) | `magnetar-component-engine` | `component`, `component_wasmtime`, `component_web`, `model_component`, `graph_builder_capability`, `model_config_capability`, `wit/` |
| Runtime orchestration (4) | `magnetar-runtime` | everything else |

A module may depend on its own layer and on layers of a strictly lower rank. The kernel pipeline and the Component engine share rank 3 and do not depend on each other.

Seven modules stay in Runtime orchestration although the audit, or the first version of this design, placed them lower: `affinity`, `tensor`, `operator_scope`, `planning`, `scheduler`, `multi_device_placement`, `kernel_optimization_orchestration`. Each depends on orchestration by design (a `dyn Provider`, a `Runtime`, the Kernel Registry, the Inference API scope check). Their data types move down to core modules instead. The layer map records the reason for each under `[held]`.

`compute` and `provider` stay in Runtime orchestration: `compute` is the Compute Capability contract and its error model, `provider` is the Provider trait and registry.

*Alternative: the audit's six crates including `magnetar-providers`.* Rejected, see the proposal.

### D2. Order of work: gate first, then layer in place, then extract

1. Inventory, layer map, layering gate with a debt list of the existing upward dependencies.
2. Remove upward dependencies inside `magnetar-runtime`; the debt list shrinks with each pull request.
3. Split `first_native_runtime` and the monolithic test files.
4. Extract crates bottom-up, one per pull request, once the debt list is empty for that crate's modules.

The gate comes first so that no new upward dependency lands while the existing ones are removed.

*Alternative: extract crates first and fix what breaks.* Rejected: the first extraction would not compile until most upward dependencies were gone, giving one unreviewable pull request.

### D3. How cycles are broken

In order of preference:

1. Move the type to the lowest module that uses it, and re-export it from its old path.
2. Replace a concrete upward reference by a trait defined in the lower layer and implemented in the upper one.
3. Move the function that causes the upward reference up to the layer it reaches.

No public item is renamed or removed. A moved item stays reachable at its old path through `pub use`.

### D4. Public path stability

`magnetar-runtime` re-exports each extracted crate's modules at their current paths (`pub use magnetar_core::tensor;` and the existing root `pub use tensor::*`). A snapshot of the public API (`cargo public-api` or an equivalent rustdoc-JSON listing) is taken before the first move and compared after each stage. The only accepted differences are additions.

Types keep their identity because they are defined once and re-exported, so trait implementations in external Providers still apply.

### D5. Roadmap modules

`provider_roadmap` and `model_format_roadmap` are leaf modules of Runtime orchestration: no module of layers 1 to 3 names them in code (the references in `kernel_artifact`, `conformance` and `production_model_ingestion` are documentation links). They stay as they are; the layering gate fails if a lower layer starts to depend on them.

### D5b. The model data layer

The ten upward dependencies that remain after the first pass share one cause. The Component engine names model data types (`ModelManifest`, `ModelArchitecture`, `ModelArchitectureConfig`, `ArtifactFormat`, `TokenizerFamily`), and the kernel pipeline names `ModelInstanceId`, `AdapterRevision` and the Model Instance selection and warmup types. These types cannot move alone: their constructors return `ModelArtifactError`, `ModelInstanceError`, `SessionError`, `AdapterError`, `TokenizerError` and `ProviderError`, and `ModelInstanceError` embeds `ModelArchitecture`.

Resolution: add a `model-data` layer of rank 2, beside graph and memory, holding the model, tokenizer, session, adapter and Model Instance *data* types with their error enums, split from the loading, residency and lifecycle code that stays in Runtime orchestration. `capability -> provider` is resolved the same way, by moving the `ProviderError` enum to core. Whether `model-data` becomes a crate of its own or part of `magnetar-core` is decided when the split is done, from its size.

### D6. Split of `first_native_runtime`

`first_native_runtime` is a directory module. `mod.rs` holds the imports, declares the submodules and re-exports them; each submodule starts with `use super::*;`.

| Submodule | Contents |
|---|---|
| `executor` | `RuntimeModelExecutor`, prefill and decode inputs and results, `FirstNativeRuntimeError`, `E2eConformanceError` and its conversions |
| `fixture` | `E2eFixture`, fixture weights and tokenizer, fixture Runtimes and instances |
| `engine` | the model execution engine, the Component graph cache, decode step replay |
| `dispatch` | per-operator dispatch through the Kernel Registry and prepared plans |
| `graph_execution` | prefill and decode graph execution, node by node |
| `kv_state` | KV update transaction: pending layers, commit, discard, release |
| `segments` | Provider runs of the graph, whole and by multi-device segment |
| `weights` | weight binding and transactional materialization |
| `prepared_plans` | `FirstNativePreparedPlans` and plan preparation |
| `production` | Component artifact resolution, `ProductionLoadedModel`, generation requests, `GenerationStreamEvent` and streaming |
| `component_graphs` | `RegisteredInferenceComponent`, `FirstNativeComponentGraphs`, graph construction |
| `chat_session` | fixture generation runs and `FirstNativeChatSession` |
| `conformance_report` | the end-to-end conformance checks and `E2eConformanceReport` |

No submodule exceeds 2,500 lines of non-test code. An item that was private to the single file is `pub(super)` in its submodule: visible in `first_native_runtime` and its children, exactly as before, and never `pub`.

The split is one scripted commit, because every moved item also needs its visibility written out; a reviewer checks it with the public API diff and the unchanged test names.

Tests move next to the code they test in a second step. `src/tests.rs` is split the same way: each test moves to the module it exercises, or to `tests/contract_tests/` when it uses only public items.

### D7. Layering gate

`tools/layering-check` parses the crate with `syn`, skips `#[cfg(test)]` code, resolves crate-root re-exports and glob imports to the module that defines each item, and reads a version-controlled layer map (`quality/layers.toml`: module → layer, held modules with reasons, and a debt list). It fails when a module depends on a higher layer or on a different layer of the same rank and the edge is not in the debt list, when a debt entry no longer matches a dependency, or when a module is missing from the map. It runs in `quality.yml` on every pull request. It is a small tool next to `tools/coverage-ratchet`, with its own unit tests. Before extraction it checks module paths. After extraction, Cargo enforces crate direction and the tool keeps checking modules inside layer 4.

### D8. Publication and versions

The four new crates are published at the same version as `magnetar-runtime`, in dependency order, by the existing release workflow. `magnetar-runtime` depends on them with exact (`=`) versions, so one release is one consistent set. Feature flags `wasmtime-component-engine` and `web-component-engine` stay on `magnetar-runtime` and forward to `magnetar-component-engine`.

Path dependencies are used in the workspace. The submodule workspace and the external repositories keep depending on `magnetar-runtime` only.

## Risks / Trade-offs

- **A cycle cannot be broken without a public change** → the module stays in layer 4 and the layer map records it. The crate set may end smaller than the table in D1.
- **Conflicts with open branches on `first_native_runtime.rs`** → the split (stage 4) starts only when no open pull request touches the file, and lands as a sequence of move-only commits.
- **`git blame` loses history on moved code** → moves and edits are separate commits; moves use `git mv` where a whole file moves.
- **Longer clean builds, or slower incremental builds, from five crates** → clean and incremental build times are measured before and after each extraction and reported in the pull request.
- **Orphan rule: an impl that was legal in one crate is illegal across two** → found at compile time during extraction; resolved by moving the impl next to the type or the trait.
- **Coverage baseline shifts when files move** → the baseline is re-measured in the same pull request as each move, with total covered lines not lower than before.
- **A partial crates.io release leaves an inconsistent set** → `tools/release-publish` publishes in dependency order and stops at the first failure; the dry-run job packages all five crates.

## Migration Plan

Each stage is one or more pull requests that leave `main` releasable:

1. Inventory, layer map, API snapshot.
2. Cycle removal (several pull requests, grouped by cycle).
3. Layering gate on.
4. `first_native_runtime` and test file split.
5. One pull request per extracted crate: core, graph, kernel, component engine.
6. Release automation, documentation, submodule pin check.

Rollback of any stage is a revert of its pull requests: no data, artifact format or contract changes.

## Open Questions

- Is `model-data` (D5b) a crate of its own or part of `magnetar-core`?
- Several moved types have `pub(crate)` constructors or fields used by higher modules (for example `ModelInstanceId::runtime_issued`). Across crates they need a public form or a different issuing path; each case is settled at extraction.
- Should the new crates be published, or kept `publish = false` with `magnetar-runtime` vendoring them? D8 assumes publication, since crates.io requires every dependency of a published crate to be published.
