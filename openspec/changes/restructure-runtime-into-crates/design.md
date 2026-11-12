## Context

`magnetar-runtime` is one crate of 69 top-level modules. Its largest file, `first_native_runtime.rs` (13,368 lines, plus 11,750 lines of tests), is the generation engine: it builds Component graphs, caches and fuses them, dispatches each node to a Provider, keeps KV state, captures and replays decode steps, materializes weights (packed or dequantized), applies layer placement and tensor-parallel plans, and implements `ProductionLoadedModel`. Providers (`providers/cpu`, `cuda`, `metal`, `npu`, `rocm`, `tpu`, `wgpu`) and Loaders are external submodules that depend on `magnetar-runtime` by path.

What the Providers actually import, measured on `providers/cuda` and `providers/cpu`: `kernel`, `reference_cpu`, `operator`, `provider`, `affinity`, `device`, `compute`, `packed_weight`, `memory`, `scheduler`, `tensor_parallel`, `layer_placement`, `conformance`, `capability`, `tensor`, and one item of `first_native_runtime`.

## Goals / Non-Goals

**Goals:**
- Make the generation engine navigable and changeable one responsibility at a time.
- Let Providers depend on a small, stable API crate instead of the whole Runtime.
- Isolate the kernel optimization pipeline and the Wasm Component host behind crate boundaries.
- Remove roadmap scaffolding from the compiled Runtime without breaking the Loaders that use part of it.

**Non-Goals:**
- Any behavior, numerics or performance change; any WIT change; removing public paths (see proposal).

## Decisions

### 1. Split inside the crate before extracting crates

Stage 1 moves `first_native_runtime.rs` into `first_native_runtime/` submodules without changing crate boundaries. It is mechanical, fully covered by the existing tests, and it is what makes stage 2 tractable: the crate extraction needs to know which engine parts use which modules, which a 13k-line file hides.

Target layout:

| Module | Responsibility |
|---|---|
| `fixture` | E2E fixture, conformance report and evidence validation |
| `component_graphs` | building graphs from named Components, rope and KV edge validation |
| `graph_cache` | `ComponentGraphCache`, `CachedGraphs`, `GraphExecutionMemo`, fusion passes |
| `dispatch` | per-node operator dispatch, output placement, pre-admission and rollback |
| `execution` | graph execution order, materialized edges, KV state and commit |
| `step_replay` | `StepReplay`, decode-step capture and replay, tracing |
| `weights` | weight materialization transaction, packed/streamed staging |
| `placement` | production layer placement and tensor-parallel plan application |
| `production` | `ProductionLoadedModel`, the generation loop, sampling hooks |

Each module gets its own `tests.rs`; `first_native_runtime/tests.rs` is split along the same lines. Shared test helpers move to a `test_support` module compiled only for tests. `first_native_runtime` keeps re-exporting every public item at its current path.

Alternative considered: extract crates first (the audit's order). Rejected: extraction across a file this size forces large, risky moves with no intermediate check.

### 2. Crate DAG: an API crate below the Runtime, not Providers above it

```
magnetar-provider-api   (kernel, operator, provider, affinity, device, compute,
        |                packed_weight, memory, scheduler, tensor, tensor_parallel,
        |                layer_placement, operator semantics shared with Reference CPU)
        v
magnetar-kernel         (kernel artifacts, manifests, ingestion, signatures,
        |                compilation, benchmark, performance model, autotuning,
        |                qualification, cache, selection, optimization orchestration)
        v
magnetar-component-host (Wasmtime and web Component hosts, Capability linking, WIT bindings)
        v
magnetar-runtime        (generation engine, model loading, inference API, sessions,
                         KV and prefix caches, scheduling; re-exports everything above)

providers/* ----depend on----> magnetar-provider-api (or magnetar-runtime, unchanged)
```

Providers implement a trait the Runtime defines; they cannot sit above it. The API crate holds exactly that trait and the types in its signatures. `memory` belongs there because `submit_kernel` takes the `MemoryManager`; the parts of `reference_cpu` Providers mirror (`MoeRouting`, `moe_local_experts`, attribute constants) move with the operator definitions, while the Reference CPU executor itself stays in the Runtime.

The exact closure is measured, not guessed: the first stage-2 task lists every item Providers import and every item those items' signatures need, and the crate holds that closure.

Alternative considered: one `magnetar-core` holding all base types (the audit's proposal). Rejected as the first cut: "core" invites unrelated types; a crate defined by the Provider contract has a clear membership rule.

### 3. Compatibility by re-export

`magnetar-runtime` keeps `pub use magnetar_provider_api::kernel;` (and so on) for every moved module, so `magnetar_runtime::kernel::KernelInvocation` and every other path still resolve. Submodules and the CLI need no change in this change. Dropping the old paths is a later, separate change that bumps versions.

### 4. Roadmap modules

`model_format_roadmap` contains two kinds of items:

- Live metadata types and validators that Loaders and Runtime code use (`HfConfigMetadata`, `HfRopeMetadata`, tokenizer, generation and chat template metadata, `SafetensorsManifest`, shard checks, `detect_duplicate_tensor_names`, quantization declarations, local-file boundary checks). These move to `model_format`, with `model_format_roadmap` re-exporting them for one release.
- Roadmap contract items (`ModelFormatRoadmapPhase`, `MODEL_FORMAT_ROADMAP_PHASES`, rejection rules for unimplemented formats). These move to `roadmap-contracts` with their tests.

`provider_roadmap` is roadmap contract only (its conformance entry point is referenced, not called, by `conformance.rs`); it moves to `roadmap-contracts` whole, and the two doc references are updated.

## Risks / Trade-offs

- [Large mechanical diffs conflict with ongoing work on the same files] -> land each stage as its own pull request, rebase-free moves (`git mv` plus path fixes), and schedule stage 1 between feature merges.
- [A moved item's privacy changes: `pub(crate)` items used across the new crate boundary] -> keep them crate-private where possible by moving their users along; widen visibility only for items in the measured Provider closure, documented as such.
- [Tests reach private engine state] -> stage 1 keeps tests as child modules of the code they test, so privacy is unchanged; only cross-module helpers move to `test_support`.
- [Build-time regressions from more crates] -> measure clean and incremental build times of `magnetar-runtime` and `providers/cuda` before and after each stage.
- [Re-exports hide the new structure] -> new code in the workspace uses the new paths; a later change removes the old ones.

## Migration Plan

1. Stage 1 (in-crate split): one pull request; all tests unchanged.
2. Stage 2a (`magnetar-provider-api`), 2b (`magnetar-kernel`), 2c (`magnetar-component-host`): one pull request each, each leaving every dependent compiling through re-exports.
3. Stage 3 (roadmap modules): one pull request in the Runtime and `roadmap-contracts`; `loaders/huggingface` unchanged.
4. Optional follow-ups, outside this change: dependents switch to new paths; a versioned change removes old paths.

Rollback: each stage is a pure move and reverts as one commit.

## Open Questions

- Whether the generation engine itself becomes a crate after stage 2 (decided once the module boundaries from stage 1 have been exercised).
- Whether `reference_cpu` (the Reference CPU executor) becomes its own crate, since several integration tests use it as an oracle.
