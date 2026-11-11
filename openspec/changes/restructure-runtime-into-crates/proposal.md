## Why

An external audit of `claude/charming-lamport-1b9m74` found `magnetar-runtime` over-concentrated, which a check of the tree confirms and sharpens:

- `magnetar-runtime/src/` holds 69 top-level modules (the audit counted 45).
- `first_native_runtime.rs` alone is 13,368 lines, with 11,750 more in `first_native_runtime/tests.rs`; `src/tests.rs` adds 6,626. One file mixes the E2E fixture, the component graph cache, step capture and replay, per-node dispatch, KV state, weight materialization, layer placement and tensor parallelism, and `ProductionLoadedModel`.
- Two roadmap modules (`model_format_roadmap.rs`, 1,586 lines; `provider_roadmap.rs`, 1,541 lines) are compiled into the Runtime and exported, although most of their content is contract scaffolding for future work. The repository already has a `roadmap-contracts` crate for exactly that.

The cost is concrete: every change to the generation engine rebuilds and re-tests one file; unrelated features collide in it; and Providers, which only need the Runtime's provider-facing API, depend on the whole Runtime crate.

The audit's own refactoring plan needs three corrections, verified in the code:

1. `first_native_runtime.rs` is the generation engine (loading, decode loop, KV state, step capture, multi-GPU placement), not hardware backends; it must not move into a providers crate.
2. Providers (`providers/cpu`, `providers/cuda`, ...) are already external submodules that depend on `magnetar-runtime` to implement its `ProviderExecutionApi`. The proposed DAG that puts `magnetar-providers` between the Wasm host and the Runtime inverts that direction. The right cut is a small provider API crate that both the Runtime and the Providers depend on.
3. `model_format_roadmap` is not dead code: `loaders/huggingface` imports `HfConfigMetadata`, `HfRopeMetadata`, tokenizer metadata types and `detect_duplicate_tensor_names` from it. Its live parts must move to a correctly named module before the roadmap scaffolding leaves.

## What Changes

Three stages, each shippable on its own, with no behavior change and no WIT change:

1. **Split `first_native_runtime.rs` into submodules inside `magnetar-runtime`**: fixture and conformance report, component graph construction and cache, per-node dispatch, graph execution and KV state, step capture and replay, weight materialization, layer placement and tensor parallelism, and the production loaded model. Tests move next to the code they test. Public paths stay valid through re-exports.
2. **Extract crates in dependency order**: first `magnetar-provider-api` (what Providers actually import: `kernel`, `operator`, `provider`, `affinity`, `device`, `compute`, `packed_weight`, `memory`, `scheduler`, `tensor`, `tensor_parallel`, `layer_placement`, and the operator semantics Providers mirror from `reference_cpu`), then `magnetar-kernel` (kernel artifacts, ingestion, compilation, benchmarking, autotuning, qualification, cache and selection), then the Wasm Component host last. `magnetar-runtime` re-exports every moved module at its old path, so Providers, Loaders, Components and the CLI keep compiling unchanged; they migrate to the new crates at their own pace.
3. **Take the roadmap modules out of the compiled Runtime**: the HF/tokenizer/shard metadata types and validators that Loaders use move to a `model_format` module (re-exported at the old path for one release); the roadmap contracts (phases, rejection rules for future formats and Providers) move to `roadmap-contracts`.

## Capabilities

### New Capabilities

- `runtime-crate-structure`: how the Runtime is split into crates and modules, the dependency direction between them, and the compatibility guarantee for public paths during the move.

### Modified Capabilities

None: no Runtime behavior, Provider contract, Component contract or WIT world changes.

## Impact

- **Owning layer**: Runtime (internal structure only). No Component, Capability, Provider or Device semantics change; Resource Affinity and Resolution Policy are untouched.
- **Providers**: keep depending on `magnetar-runtime` unchanged; may switch to `magnetar-provider-api` later for faster, smaller builds.
- **Loaders**: `loaders/huggingface` keeps compiling through the `model_format_roadmap` re-export; it switches to `model_format` when convenient.
- **WIT**: no change to any WIT package or world.
- **Public API**: every public item keeps its current path through re-exports; new crate paths are additive. Removing the old paths is a separate, later change.
- **Build**: the root workspace gains members; `submodule-workspace` and submodule `Cargo.toml` path dependencies stay valid.

## Non-goals

- No behavior, numerics, performance or scheduling change; any such change belongs to its own proposal.
- No change to WIT contracts, Component authority or Capability linking.
- No new Provider responsibilities and no movement of hardware code into the Runtime, or of the generation engine into Providers.
- No removal of public paths in this change.
- `magnetar-cli` (tests for `process.rs` timeouts, serve mode) is out of scope; the audit's CLI findings are tracked separately.
- No crate for the generation engine itself yet: it stays in `magnetar-runtime`, split into modules; whether it deserves its own crate is decided after stage 2.
