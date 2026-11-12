## Why

The project audit of 2026-11 rates every axis of Magnetar as solid except internal modularity. `magnetar-runtime` is one crate of 66 top-level modules and about 146,500 lines. Three files concentrate the problem:

- `first_native_runtime.rs`: 12,980 lines and 75 public items. It mixes the model executor contract, the first-native fixture, KV state, multi-device segments, prepared plans, production generation, streaming, Component graph construction, chat sessions and the E2E conformance report.
- `first_native_runtime/tests.rs` (11,600 lines) and `src/tests.rs` (6,542 lines): they reach private items through `use super::*`, so any move of production code drags them along.

The audit proposes six crates in a strict DAG. Measured against the code, that target cannot be reached by moving files, because the modules are not layered today. A syntax-level dependency analysis (the `layering-check` tool this change adds) found, on `main` after astorise/Magnetar#167:

- 55 of the 69 top-level modules in one dependency cycle.
- 68 dependencies pointing from a lower layer to a higher one, against the audit's layering. More than half go to `compute`, which defines the data types every layer uses (`ComputeDType`, `TensorDescriptor`, `ShapeDescriptor`, `TensorResourceId`, `ComputeValidationError`) next to the Compute Capability contract itself.
- The same pattern in `affinity` (bindings and status types next to `dyn Provider` resolution), `scheduler` (Provider execution errors), `provider` (ABI version), `tensor`, `observability` and `model_instance`.
- Several modules that the audit places low depend on orchestration by design: `scheduler` schedules on a `Runtime`, `planning` plans a `ComputeGraph` against Provider metadata, `operator_scope` checks graphs against the Kernel Registry.

This change belongs in Magnetar because it only reorganises code the Runtime already owns. It adds no responsibility and moves none across the Component, Runtime, Provider or Device boundaries.

## What Changes

- Define five layers for Runtime code and the allowed dependency direction between them: core types, graph and memory, kernel pipeline, Component engine, Runtime orchestration.
- Remove the upward dependencies, so that each module depends only on its own layer or a lower one. Shared data types move down to core modules and stay re-exported from their old paths.
- Split `first_native_runtime.rs` into submodules by responsibility. Split its test file and `src/tests.rs` so that each test module sits next to the code it tests.
- Keep `provider_roadmap` and `model_format_roadmap` as leaf modules: no lower layer depends on them today, and the gate keeps it so.
- Add a CI gate that fails when a module depends on a higher layer. The upward dependencies that remain are listed as debt in the layer map; the gate rejects new ones and stale entries, so the list only shrinks.
- Extract the four lower layers into workspace crates: `magnetar-core`, `magnetar-graph`, `magnetar-kernel`, `magnetar-component-engine`. `magnetar-runtime` keeps Runtime orchestration and re-exports the four crates, so every existing `magnetar_runtime::…` path still resolves.
- Extend release publication, coverage and documentation gates to the new crates.

### Where this change departs from the audit

| Audit recommendation | Decision | Reason |
|---|---|---|
| A `magnetar-providers` crate holding "CPU & CUDA backends", fed by `first_native_runtime.rs` | Not created | Providers already live outside this repository as pinned submodules (`providers/cpu`, `providers/cuda`, …). `first_native_runtime.rs` holds Runtime orchestration and conformance code, with no Provider implementation. `reference_cpu` stays in the Runtime as the correctness baseline (`provider-roadmap` spec). |
| A crate named `magnetar-wasm-host` | Named `magnetar-component-engine` | `Host` is not a primary Magnetar concept (`project-architecture` spec). |
| Delete the "transitional" roadmap modules | Kept as leaf modules | `roadmap-contracts` and the external Formats and Loaders use their public items, and the `provider-roadmap` and `model-format-roadmap` specs are live. Inside the Runtime only `model_format_roadmap` uses `provider_roadmap`. |
| Graph and kernel as sibling layers, `affinity`, `planning`, `scheduler`, `tensor` in low layers | Kernel pipeline sits above graph and memory; seven modules are held in Runtime orchestration | The kernel pipeline names `ExecutionGraph` and memory identifiers. The held modules depend on orchestration by design; their data types move down instead (`quality/layers.toml`, `[held]`). |
| `magnetar-cli` has no tests | Not acted on here | The crate has 93 unit tests across its 13 modules. See Non-goals. |

## Capabilities

### New Capabilities

- `runtime-modularity`: layering of Runtime code, dependency direction, public path stability, and the size and responsibility of first-native modules.

### Modified Capabilities

- `quality`: a layering gate, and coverage and publication checks that span the Runtime crates.

## Impact

- `magnetar-runtime`: every module moves or changes its imports. No behaviour change, no new public item, no removed public item.
- New crates `magnetar-core`, `magnetar-graph`, `magnetar-kernel`, `magnetar-component-engine` in the root workspace.
- Consumers (`magnetar-cli`, `inference-components`, `roadmap-contracts`, the three integration test crates, and the 13 Component, Provider, Format and Loader repositories): no source change required. They are rebuilt and tested against the result.
- `.github/workflows/quality.yml`, `release-publication.yml`, `quality/coverage-baseline.json`, `tools/coverage-ratchet`, `tools/release-publish`: paths and crate lists.
- WIT contracts: unchanged. The `wit/` directory moves with the Component engine, and package names, interfaces and versions stay identical.
- Authority and external access: unchanged. No filesystem, network, secret or process authority is added or moved.
- Open branches that touch `first_native_runtime.rs` (for example the quantized GGUF work in astorise/Magnetar#167) will conflict with the split. The split is sequenced after they merge.

## Non-goals

- Any behaviour, performance or numerical change. Logits, reports and error values stay identical.
- New Providers, a Provider crate in this repository, or moving `reference_cpu` out of the Runtime.
- Changing the vocabulary or the contents of the roadmap catalogs.
- The audit's `magnetar-cli` test plan. Three of its four items test features that do not exist: `magnetar serve` binds no HTTP listener and has no SSE stream, `process.rs` has no timeout or signal handling, and the CLI does not use `tracing`. They belong with `connect-production-cli-and-service`, which builds that surface. The agent step bound is already tested.
- The audit's resilience tests (KV memory saturation under concurrent batching, corrupted Kernel Bundles). They are new behaviour coverage, with no link to module structure.
- Splitting `magnetar-cli` or the external repositories.
