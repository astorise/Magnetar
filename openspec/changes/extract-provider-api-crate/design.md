## Context

After `layer-and-split-runtime-crate`, the Runtime is six crates in five layers below Runtime orchestration: core (1), graph and memory (2), model data (3), kernel pipeline and Component engine (4), Runtime orchestration (5). `quality/layers.toml` and `tools/layering-check` enforce the direction.

Providers are external repositories pinned as submodules. Each depends on `magnetar-runtime` by path and version, and commits its own `Cargo.lock`.

What keeps the Provider contract in the orchestration crate today, from `layering-check --report`:

- `provider` uses data types defined in `scheduler` (execution request, handle, result, status), `compute` (advertisement and support types, `ComputeError`), `tensor`, `packed_weight`, `layer_placement`, `tensor_parallel`.
- `scheduler` holds those data types next to code that schedules on a `Runtime`.
- `ProviderExecutionRequest` embeds `ComputeExecutionPlan`, `MemoryPlan`, `ExecutionStep` and `ExecutionConstraint`, defined in `planning` next to the planner.
- `packed_weight` and `model_loading` depend on each other: packed weights call the dequantizers, and Model Loading builds packed weights.
- `affinity` holds `AffinityResolution`, which borrows a `dyn Provider`.

## Goals / Non-Goals

**Goals:**

- A Provider can be built against `magnetar-provider-api` alone, with `magnetar-runtime` only as a dev-dependency for conformance tests.
- A change to Runtime orchestration, the Component engine or the generation engine changes no Provider `Cargo.lock`.
- Every existing `magnetar_runtime::…` path keeps resolving to the same item.

**Non-Goals:**

- Any change to the Provider contract or its behaviour.
- A stability policy for the new crate.

## Decisions

### D1. One new layer, one new crate

`provider-api` gets rank 5; Runtime orchestration moves to rank 6. `magnetar-provider-api` depends on `magnetar-core`, `magnetar-graph` and `magnetar-kernel`. It does not depend on `magnetar-engine`: Providers never see Component types. It depends on `magnetar-model` only if the inventory (task 1.1) shows a Provider-facing type that needs model data; none of `provider`, `tensor`, `packed_weight`, `tensor_parallel` or `layer_placement` names one today.

*Alternative: put the Provider contract in `magnetar-kernel`.* Rejected: the kernel pipeline does not need the `Provider` trait, and Providers would then depend on kernel ingestion, compilation and autotuning code they do not use.

*Alternative: one crate per concern (`provider-api`, `tensor`, `quantization`).* Rejected for now: eleven repositories would track more crates, which is the coupling this change removes.

### D2. Contents

| Module in `magnetar-provider-api` | Comes from | Contents |
|---|---|---|
| `provider` | `provider` | `Provider`, `ProviderExecutionApi`, `ProviderMetadata`, `ProviderDescriptor`, `ProviderRegistry`, `TensorValue`, `TensorValueAdmissionError`, `StepCaptureRequest`, `StepCaptureId`, `StepReplayError`, `PROVIDER_API_VERSION` |
| `execution` | `scheduler`, `planning` | `ProviderExecutionRequest`, `ProviderExecutionHandle`, `ProviderExecutionResult`, `ProviderExecutionStatus`, `ProviderExecutionId`, `ProviderCancellationOutcome`, `SchedulingState`; `ComputeExecutionPlan`, `MemoryPlan`, `ExecutionStep`, `ExecutionConstraint`, `ExecutionPlanId` and the types they embed |
| `compute_support` | `compute` | `ProviderComputeAdvertisement`, the operation and data movement support types, `ComputeError` and its code, phase, severity and diagnostic types |
| `affinity_resolution` | `affinity` | `AffinityResource`, `AffinityResolution` |
| `tensor`, `packed_weight`, `tensor_parallel`, `layer_placement` | same names | whole modules |
| `operator_semantics` | `reference_cpu` | `MoeRouting`, the MoE expert functions and attribute names, `row_scale`, `sigmoid`, `NATIVE_CPU_PROVIDER_NAME` |
| `quantization` | `model_loading` | the quantization tables and decoders (`iq_grids`, `iq_signs`, `IQ4_NL_VALUES`, `q3_k_scales`, `iq4_xs_scales`, `f16_to_f32`, the `dequantize_*` functions) |

Stays in `magnetar-runtime`: `ProviderLoader`, `ProviderLoadingMode`, `ProviderLoadingPolicy` (filesystem access), the scheduler, the planner, the Compute Capability contract, `ReferenceCpuProvider`, Model Loading, and the Provider conformance suite.

The table is the target. The inventory fixes the exact item list, in particular which types `ComputeExecutionPlan` and `ProviderMetadata` pull with them.

**As built.** The inventory (a closure over the identifiers each moved item names) gave a smaller set than the table. `affinity_resolution` is not needed: the Provider contract names neither affinity type. Three modules were added: `runtime_event` (the observation event types that an execution plan and a Provider result carry), `provider_api` (the contract half of `provider`, with `provider` kept as a facade) and `provider_loading` (the loader half, in Runtime orchestration). The `ComputeError` family and `memory_pressure_diagnostic` moved with `compute_support`. `reference_cpu::dtype_conversion` in the inventory is an unrelated associated function with the same name.

### D3. Items that need a decision at move time

- **`ProviderExecutionRequest::from_operation(&ScheduledOperation)`**. An inherent method cannot be defined outside the type's crate. Either `ScheduledOperation` (a data struct: id, plan, state, order, diagnostics, result) moves down with its embedded types, or the method stays reachable through a public extension trait in `magnetar-runtime`. Moving the struct is preferred if its embedded types are data only: it keeps the method path unchanged.
- **Crate-private items** used across the new boundary are handled as in `layer-and-split-runtime-crate` design D9: an item that carries no authority may become `#[doc(hidden)] pub`; one that seals an authority stays private and the caller moves or uses the public API.
- **`ProviderRegistry`** has private fields and is written by Providers through `register`. It moves whole; nothing in it is widened.

### D4. Compatibility

`magnetar-runtime` re-exports each moved module and item at its current path, as it does for the five existing layer crates. Types are defined once, so a Provider built against `magnetar-provider-api` and a Runtime re-exporting the same crate agree on the `Provider` trait and every type it names.

Verification is the one used for the crate split: a probe crate importing every public path of the pre-change `magnetar-runtime` listing must compile, the same test functions must exist and pass, and every consumer builds without source change before the Providers switch.

### D5. Versions

`magnetar-provider-api` is published with the Runtime crates at the same version, and `magnetar-runtime` depends on it with an exact version, like the other layers.

Providers depend on `magnetar-provider-api` with a caret requirement on the major version. Whether the layer crates under it keep exact requirements between themselves is decided here from the observed effect: exact requirements force a Provider lockfile update on every release of any layer; caret requirements do not.

### D6. Switching the Providers

After the crate is on `main`, each Provider repository gets one pull request: `magnetar-provider-api` replaces `magnetar-runtime` in `[dependencies]`, `magnetar_runtime::` paths become `magnetar_provider_api::` (or the lower layer crate's), `magnetar-runtime` moves to `[dev-dependencies]` where the conformance suite or the Reference CPU oracle is used, and the release workflow checks the new crate on crates.io. Submodule pins in Magnetar follow.

A Provider that has not switched keeps building: it still depends on `magnetar-runtime`, which re-exports everything.

### D7. Where the crate lives

`magnetar-provider-api` stays in the Magnetar workspace, next to the five other layer crates, and is published with them. It is not placed in a repository of its own pinned as a submodule.

Two reasons. First, the crate depends on `magnetar-core`, `magnetar-graph` and `magnetar-kernel`, and `magnetar-runtime` depends on it. In a separate repository, one release of Magnetar would be published in two rounds with a round trip between the repositories (core, graph and kernel; then the API crate; then the Runtime), which is the release coupling this change is meant to remove for Providers. Second, the root workspace deliberately has no submodule among its members, so that a plain `cargo build` works in a checkout without submodules (see the comment in `Cargo.toml`); a Runtime that depends on a submodule crate would end that.

What Providers gain does not depend on the location: they depend on one small crate whose contents change rarely, so Runtime changes above it stop reaching their lockfiles. A separate repository could be reconsidered together with moving `magnetar-core`, `magnetar-graph` and `magnetar-kernel` there, which gives a single direction (contract repository first, Magnetar second); that is a larger change than this one and is not proposed here.

## Risks / Trade-offs

- **A Provider-facing type turns out to depend on orchestration** → it stays in `magnetar-runtime`, the layer map records why, and the Provider that needs it keeps `magnetar-runtime` as a normal dependency until a follow-up.
- **Two paths to the same item** (`magnetar_runtime::provider::Provider` and `magnetar_provider_api::provider::Provider`) → both are the same type; documentation states that Providers use the API crate and applications use `magnetar-runtime`.
- **The contract looks stable because it has its own crate** → the crate documentation and this change's non-goals state that no stability policy is attached yet.
- **Seven more pull requests and pin updates** → one time. After it, Runtime changes above the API no longer touch Provider repositories.
- **Coverage moves between crates** → re-measured in the same pull request, with no exclusion added.

## Migration Plan

1. Inventory: item-level list of what moves, from `layering-check --report` and the Provider sources.
2. Move data types down inside `magnetar-runtime` (execution, plans, compute support, operator semantics, quantization), with the layer map extended by a `provider-api` layer and the gate kept green.
3. Extract `magnetar-provider-api`; `magnetar-runtime` re-exports it.
4. Release automation and documentation.
5. One pull request per Provider repository, then the submodule pins.

Each step leaves `main` releasable. Rollback is a revert: no data, artifact format or contract changes.

## Open Questions

- Does `ScheduledOperation` move with `ProviderExecutionRequest` (D3)?
- Exact or caret version requirements between the layer crates (D5)?
- Should Formats and Loaders get a similar small dependency later (`magnetar-model` plus the ingestion types)? Out of scope here.
