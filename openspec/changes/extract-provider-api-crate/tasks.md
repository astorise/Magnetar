## 1. Inventory

- [ ] 1.1 List, item by item, what the seven Provider repositories import from `magnetar-runtime`, and where each item is defined today (`layering-check --report` plus the Provider sources). Separate non-test code from tests.
- [ ] 1.2 Fix the list of items that move (design D2), including the types embedded by `ComputeExecutionPlan`, `MemoryPlan` and `ProviderMetadata`. Record any Provider-facing item that cannot move, with the dependency that keeps it.
- [ ] 1.3 Decide the open questions of the design: `ScheduledOperation` (D3), version requirements between layer crates (D5).
- [ ] 1.4 Record the baseline: public path listing of `magnetar-runtime`, test function names, and `cargo test --locked` result in each Provider repository.

## 2. Move Provider-facing data types down (inside `magnetar-runtime`)

- [x] 2.1 Add the `provider-api` layer to `quality/layers.toml` (rank above the kernel pipeline and the Component engine, below Runtime orchestration). Done: layer `provider-api`, rank 5; Runtime orchestration is rank 6.
- [x] 2.2 `scheduler` and `planning`: move the execution request, handle, result and status types and the execution plan and memory plan types to an `execution` module; re-export from their old modules. Done: the execution and memory plan types of `planning` and the Provider execution request, handle, progress, status, result, scheduled operation and scheduler error types of `scheduler` are in `execution`, with `runtime_events_for_execution_plan`; `planning` and `scheduler` re-export them. The event types a plan and a result carry (`RuntimeEvent`, `RuntimeEventKind`, `RuntimeObservationPhase`, `RuntimeDiagnosticCode`) and the observation sequence counter moved to a new `runtime_event` module, re-exported from `observability`.
- [x] 2.3 `compute`: move the Provider compute advertisement, support and error types to `compute_support`; re-export. Done: the advertisement, operation schema, support and data-movement types of `compute` are in `compute_support`, with the `ComputeError` family (`ComputeError`, its code, phase, severity, diagnostic and recovery hint) and `memory_pressure_diagnostic`, which the plan and Provider code build on. The conversions into `ComputeError` from `ComputeValidationError` and `ComputePlanningError` stay beside those types, in the Runtime-layer modules (legal across crates: the source type is local).
- [x] 2.4 `affinity`: move `AffinityResource` and `AffinityResolution` to `affinity_resolution`; re-export. Not needed: the closure of the Provider contract names neither `AffinityResource` nor `AffinityResolution`, so `affinity` stays in Runtime orchestration.
- [x] 2.5 `reference_cpu`: move the operator semantics Providers reuse to `operator_semantics`; re-export. Done: `NATIVE_CPU_PROVIDER_NAME`, `sigmoid`, `row_scale`, `moe_route`, `MoeRouting`, `moe_experts`, `moe_experts_local`, `moe_local_experts`, the two expert attribute names and the shared `rows_cols_of` helper are in `operator_semantics`; `reference_cpu` re-exports the public ones. 950 test names unchanged, 1,703 tests pass, probe compiles.
- [x] 2.6 `model_loading`: move the quantization tables and decoders to `quantization`; re-export. `packed_weight` no longer depends on `model_loading`. Done: the 15 block-size constants, `K_SCALE_SIZE`, `IQ4_NL_VALUES`, the twelve `dequantize_*` decoders with their helpers, the scale unpackers, the `iq_grids` tables and the four half-precision conversions (`f16_to_f32`, `bf16_to_f32`, `f32_to_f16`, `f32_to_bf16`) are in `quantization`; `model_loading` re-exports the items that were public. `packed_weight` no longer names `model_loading`. 950 test names unchanged, 1,703 tests pass, probe compiles.
- [x] 2.7 Split `provider`: the contract (traits, metadata, registry, tensor value, step capture) is assigned to the `provider-api` layer; `ProviderLoader` and the loading policy move to a `provider_loading` module in Runtime orchestration and stay re-exported from `provider`. Done: `provider` is split into `provider_api` (traits, metadata, descriptor, registry, tensor value, step capture) in the `provider-api` layer and `provider_loading` (`ProviderLoader`, loading mode and policy, the ABI factory symbol) in Runtime orchestration; `provider` remains as a facade that re-exports both, so every path is unchanged. The seven loader tests moved to `provider_loading/tests.rs`.
- [x] 2.8 Assign `tensor`, `packed_weight`, `tensor_parallel` and `layer_placement` to the `provider-api` layer. Layering gate passes with no debt entry. Done: `tensor`, `packed_weight`, `tensor_parallel` and `layer_placement` are in the `provider-api` layer; the gate passes on 79 modules with no debt entry.
- [x] 2.9 After each task above: same test functions as the baseline, all passing; Clippy and documentation gates clean. Done at each step: same 950 test function names as the baseline, 1,703 workspace tests passing with default and all features, Clippy and documentation gates clean, the public path probe compiling.

## 3. Extract the crate

- [ ] 3.1 Create `magnetar-provider-api` in the root workspace with the modules of the `provider-api` layer; `magnetar-runtime` re-exports each module and item at its former path.
- [ ] 3.2 Handle crate-private items used across the boundary as in design D3; list each widened item and why it carries no authority.
- [ ] 3.3 Verify: the public path probe compiles against the new `magnetar-runtime`; the dependency graph of `magnetar-provider-api` contains neither `magnetar-runtime` nor `magnetar-engine`; the crate builds for `wasm32-unknown-unknown`.
- [ ] 3.4 Build and test the CLI, `inference-components`, the integration test crates and the eleven pinned repositories without source change.
- [ ] 3.5 Build each crate on Linux, macOS and Windows (CI matrix).

## 4. Release, gates and documentation

- [ ] 4.1 `release-publication.yml`: package and publish `magnetar-provider-api` with the other Runtime crates; dry run covers it.
- [ ] 4.2 `quality.yml`: add the crate to the layering, model-family isolation and test-placement jobs. Add a job that fails if `magnetar-provider-api` depends on `magnetar-runtime` or `magnetar-engine`.
- [ ] 4.3 Coverage: re-measure; no exclusion added for moved code.
- [ ] 4.4 Update `README.md`, `CONTRIBUTING.md`, `SUBMODULES.md`, the Provider documentation under `docs/` and `openspec/config.yaml`: Providers depend on `magnetar-provider-api`, applications on `magnetar-runtime`.
- [ ] 4.5 `CHANGELOG.md` entry.

## 5. Switch the Provider repositories

- [ ] 5.1 Reference CPU oracle and conformance suite: confirm which Providers need `magnetar-runtime` as a dev-dependency.
- [ ] 5.2 One pull request per Provider repository (CPU, CUDA, Metal, NPU, ROCm, TPU, WGPU): depend on `magnetar-provider-api`, update `use` paths and `Cargo.lock`, update the release workflow's upstream check. `cargo test --locked` passes with the same tests as the baseline.
- [ ] 5.3 Update the seven submodule pins in Magnetar.
- [ ] 5.4 Verify the requirement "Runtime Changes Do Not Reach Providers": add a dependency to `magnetar-runtime` on a scratch branch and confirm that `cargo metadata --locked` still succeeds in each Provider repository.
- [ ] 5.5 Run GPU CI (`cuda-first-native`, `multi-device-cpu-cuda`): results identical to the baseline.
