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

| Layer | Crate | Modules (target) |
|---|---|---|
| 1. Core types | `magnetar-core` | `capability`, `device`, `tensor`, `operator`, `operator_scope`, `resolution`, `affinity`, `artifact_signature` |
| 2. Graph and memory | `magnetar-graph` | `execution_graph`, `graph_fusion`, `execution_stream`, `planning`, `memory`, `scheduler`, `multi_device_placement` |
| 2. Kernel pipeline | `magnetar-kernel` | the 15 `kernel*` modules |
| 3. Component engine | `magnetar-component-engine` | `component`, `component_wasmtime`, `component_web`, `model_component`, `graph_builder_capability`, `model_config_capability`, `wit/` |
| 4. Runtime orchestration | `magnetar-runtime` | `runtime`, `provider`, `compute`, `reference_cpu`, `model*`, `inference_api`, `generation`, `sampling`, `tokenizer`, `session`, `kv_cache`, `prefix_cache`, `batching`, `adapter`, `first_native*`, `conformance`, `cli_boundary`, `observability`, `phase_attribution`, `production_model_ingestion`, roadmap modules |

A module may depend on its own layer and on lower layers. Graph and kernel are both layer 2 and do not depend on each other, unless the cycle inventory (task 1.1) shows a one-way dependency, in which case the direction is recorded in the layer map.

The table is the target. Where the inventory shows that a module cannot sit in its target layer without changing a public type, the module stays in layer 4 and the layer map records why. `compute` and `provider` are the likely cases: both are used by almost every module, and both reach `inference_api`.

*Alternative: the audit's six crates including `magnetar-providers`.* Rejected, see the proposal.

### D2. Order of work: layer in place, then extract

1. Inventory and layer map.
2. Break cycles inside `magnetar-runtime`.
3. Turn on the layering gate.
4. Split `first_native_runtime` and the monolithic test files.
5. Extract crates bottom-up, one per pull request.

*Alternative: extract crates first and fix what breaks.* Rejected: the first extraction would not compile until most cycles were gone, giving one unreviewable pull request.

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

`provider_roadmap` and `model_format_roadmap` become leaf modules of layer 4: nothing in layers 1 to 3 depends on them. The types that `kernel_artifact`, `conformance` and `production_model_ingestion` use move to their owning modules and are re-exported from the roadmap modules.

### D6. Split of `first_native_runtime`

`first_native_runtime` becomes a directory module. Indicative submodules, following the existing public types:

| Submodule | Contents |
|---|---|
| `executor` | `RuntimeModelExecutor`, prefill and decode inputs and results, `FirstNativeRuntimeError` |
| `fixture` | `E2eFixture`, `FirstNativeFixtureGeneration` |
| `kv_state` | `FirstNativeLayerKvState` and commit, discard, release |
| `segments` | multi-device segment boundaries and provider run outcomes |
| `prepared_plans` | `FirstNativePreparedPlans` and Kernel registration |
| `production` | `ProductionGenerationRequest`, `ProductionLoadedModel`, the `run_production_qwen_generation*` entry points |
| `streaming` | `GenerationStreamEvent` |
| `component_graphs` | `RegisteredInferenceComponent`, `FirstNativeComponentGraphs` |
| `chat_session` | `FirstNativeChatSession` |
| `conformance_report` | `E2eConformanceError`, `E2eTestStatus`, `E2eTestResult`, `E2eConformanceReport` |

`first_native_runtime/mod.rs` only declares and re-exports. No submodule exceeds 2,500 lines of non-test code. Tests move with the code they test. Items that tests reach through `super::*` become `pub(super)` or `pub(crate)`, never `pub`.

`src/tests.rs` is split the same way: each test moves to the module it exercises, or to `tests/contract_tests/` when it uses only public items.

### D7. Layering gate

A check in the root workspace reads a version-controlled layer map (`quality/layers.toml`: module or crate → layer) and fails when a `crate::…` path or a crate dependency points to a higher layer, or when a module is missing from the map. It runs in `quality.yml` on every pull request. It is a small tool next to `tools/coverage-ratchet`, with its own unit tests. Before extraction it checks module paths. After extraction, Cargo enforces crate direction and the tool keeps checking modules inside layer 4.

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

- Do `compute` and `provider` end in a lower layer, or stay in layer 4? Decided by the inventory in task 1.1.
- Should the new crates be published, or kept `publish = false` with `magnetar-runtime` vendoring them? D8 assumes publication, since crates.io requires every dependency of a published crate to be published.
