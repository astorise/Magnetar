## Why

The seven Provider repositories depend on the whole of `magnetar-runtime`, although they use a small part of it. Measured on their sources, they import about 110 items from 16 Runtime modules:

- 61 of them already live in the layer crates extracted by `layer-and-split-runtime-crate`: Kernel types (17), operators (9), Memory Manager types (9), bindings and Provider status (7), compute data types (7), devices (6), capabilities (2), and four execution error and identifier types.
- 46 are still defined in `magnetar-runtime`: the `Provider` and `ProviderExecutionApi` traits with their metadata, registry, tensor value and step capture types (11), execution request, handle, result and status types in `scheduler` (7), operator semantics the Providers mirror from `reference_cpu` (9), quantization tables in `model_loading` (6), `packed_weight` (3), `tensor` (4), `tensor_parallel` (2), `layer_placement` (1), and the Provider conformance suite (4, used by one Provider's tests).

The cost showed during `layer-and-split-runtime-crate`: adding five crates under `magnetar-runtime` changed the `Cargo.lock` of eleven external repositories, one pull request each, although no Provider source changed. Any change to the generation engine or its dependencies does the same, and rebuilds every Provider.

A small crate holding what Providers implement and call removes that coupling: a Provider then depends on the contract it implements, and Runtime internals can change without touching Provider repositories.

This belongs in Magnetar because the Provider contract is defined by the Runtime: Providers implement Capabilities and expose Devices, and the Runtime owns resolution. Only the location of the contract's Rust types changes.

## What Changes

- Add a layer `provider-api`, between the kernel pipeline and Runtime orchestration, built as a new crate `magnetar-provider-api`.
- Move into it the Provider-facing contract: the `Provider` and `ProviderExecutionApi` traits, `ProviderMetadata`, `ProviderDescriptor`, `ProviderRegistry`, `TensorValue`, the step capture types, `PROVIDER_API_VERSION`, and the modules `tensor`, `packed_weight`, `tensor_parallel` and `layer_placement`.
- Move down the data types those traits carry and that are still next to orchestration code: the execution request, handle, result and status types of `scheduler`; the execution plan and memory plan types of `planning`; the compute advertisement, support and error types of `compute`; the affinity resolution types of `affinity`.
- Move the operator semantics that Providers reuse (MoE routing and expert selection, `row_scale`, `sigmoid`, their attribute names) from `reference_cpu` to an `operator_semantics` module, and the quantization tables and decoders from `model_loading` to a `quantization` module. This also removes the `packed_weight` ↔ `model_loading` cycle.
- `magnetar-runtime` re-exports the new crate's modules and items at their current paths.
- Provider repositories switch their dependency from `magnetar-runtime` to `magnetar-provider-api`, one pull request each, after the crate exists. `magnetar-runtime` stays as a dev-dependency where a Provider runs the conformance suite.

## Capabilities

### New Capabilities

- `provider-api-crate`: what the Provider API crate contains, what it may depend on, and the compatibility guarantee for Providers and for existing `magnetar_runtime` paths.

### Modified Capabilities

None. No Provider contract, Capability, Device, Resource Affinity or Resolution Policy semantics change.

## Impact

- **Owning layer**: Runtime (location of the Provider contract types). Providers keep implementing the same Capabilities, exposing the same Devices and owning the same native responsibilities (kernels, allocators, streams, driver APIs). Resource Affinity behaviour is unchanged.
- **Depends on** `layer-and-split-runtime-crate` being merged: this change adds one layer and one crate to its layer map and gate.
- **`magnetar-runtime`**: `provider`, `scheduler`, `planning`, `compute`, `affinity`, `reference_cpu` and `model_loading` lose their data types and keep their behaviour. No public item removed or renamed.
- **Provider repositories** (CPU, CUDA, Metal, NPU, ROCm, TPU, WGPU): `Cargo.toml`, `Cargo.lock` and `use` paths, no behaviour change. Until they switch, they keep building against `magnetar-runtime` unchanged.
- **Formats and Loaders**: not affected. They use model, tokenizer and ingestion types, not the Provider contract.
- **Release**: one more crate to publish, before `magnetar-runtime`. Provider release workflows check `magnetar-provider-api` on crates.io instead of `magnetar-runtime`.
- **WIT contracts**: unchanged. Components never see Provider types.
- **Authority and external access**: unchanged. `ProviderLoader` (loading Provider libraries from the filesystem) stays in `magnetar-runtime`; the API crate has no filesystem, network, secret or process access.

## Non-goals

- Changing the Provider contract: no trait method, signature, default implementation or ABI version changes.
- Stabilising the contract as a public long-term API. This change makes it a separate crate; a stability policy for it is a later decision.
- Moving `ReferenceCpuProvider` out of the Runtime, or the Provider conformance suite out of it: the suite drives a `Runtime`.
- A separate crate for Formats and Loaders (model, tokenizer and ingestion types).
- Moving `provider_roadmap` or `model_format_roadmap`.
- Removing any `magnetar_runtime::…` path.
