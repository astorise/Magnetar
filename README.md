# Magnetar

Magnetar is a Rust runtime for portable local AI execution.

The current implementation is a v0.1 local-runtime baseline with real
Provider-backed first-native execution. The repository contains the
`magnetar-runtime` and `magnetar-cli` crates, executable contract and
integration tests, and the OpenSpec history that defines the architecture as
it grows.

Reference CPU execution is implemented, and an external CUDA Provider is now
integrated and validated through real-hardware first-native end-to-end tests.
Magnetar is **not yet a general production model runtime**: caller-facing Qwen
execution is still centered on the built-in `qwen-test` path, and arbitrary
production model artifact loading (for example general `config.json` +
Safetensors/tokenizer inputs) remains incomplete.

## Architecture

The canonical architecture is:

```text
Component
    |
    | imports Capability
    v
Runtime
    |
    | Resolution Policy plus Resource Affinity
    v
Provider
    |
    v
Device
```

- **Runtime** owns local-node orchestration, Capability resolution, Provider
  registration, Device discovery coordination, scheduling, planning,
  observability, and recovery policy.
- **Component** is portable WebAssembly Component Model code. Components use
  WIT contracts and do not receive native handles, raw pointers, queues,
  streams, Provider handles, or Device handles.
- **Capability** is a portable WIT contract describing an ability available to
  Components.
- **Provider** is a trusted native Runtime extension that implements one or
  more Capabilities and owns native implementation details.
- **Device** is a physical or logical execution target exposed by a Provider.
- **Resource Affinity** records authoritative bindings for live resources,
  artifacts, execution contexts, Providers, Devices, and future model state.
- **Resolution Policy** selects among compatible execution candidates after
  mandatory compatibility and affinity constraints have been applied.

Vendor-specific execution belongs in Providers, not in `magnetar-runtime`.
For example, CUDA allocation, kernels, streams, device buffers, and other CUDA
backend details live in the external CUDA Provider; the Runtime deals only in
generic Provider, Device, Kernel, Tensor Resource, placement, and affinity
contracts.

The canonical conceptual entry point is
[docs/architecture/overview.md](docs/architecture/overview.md).

Detailed architecture notes:

- [Capability taxonomy](docs/architecture/capability-taxonomy.md)
- [Resource affinity](docs/architecture/resource-affinity.md)
- [Resolution policy](docs/architecture/resolution-policy.md)
- [Provider health](docs/architecture/provider-health.md)
- [Compute graph submission](docs/architecture/compute-graph-submission.md)
- [Compute execution planning](docs/architecture/compute-execution-planning.md)
- [Memory planning](docs/architecture/memory-planning.md)
- [Scheduler](docs/architecture/scheduler.md)
- [Runtime observability](docs/architecture/runtime-observability.md)

## Current Status

Implemented today:

- `magnetar-runtime`
- `magnetar-cli`
- Runtime lifecycle
- Runtime configuration
- Capability, Provider, Device, Resource Affinity, and Resolution Policy models
- WebAssembly Component registration, contract validation, fail-closed import
  authorization, lifecycle management, and a feature-gated Wasmtime Component
  Engine adapter
- Memory planning and Tensor Resource contracts
- Operator, Kernel, Kernel Registry, Kernel Dispatch, and generic Provider
  execution contracts
- Reference CPU Provider execution
- External CUDA Provider baseline, with CUDA implementation details kept out of
  `magnetar-runtime`
- Model Artifact, Model Loading, Model Instance, Tokenizer, Generation,
  Sampling, Session, KV Cache, Prefix Cache, Continuous Batching, Runtime
  Inference API, and E2E conformance contract surfaces
- First-native Qwen graph execution driven by the compiled Qwen Component,
  `ExecutionGraph`, real `ModelInstance` placement, and published
  `PreparedExecutionPlan` bindings
- Provider-resolved weight materialization, Runtime-owned Tensor residency and
  Resource Affinity, and transactional KV/resource lifecycle
- Device-resident CUDA first-native chaining for supported kernels, including
  MatMul, RMSNorm, and multi-head/GQA-aware RoPE, without introducing CUDA code
  into the Runtime Core
- Real-hardware CUDA integration tests that execute the actual first-native
  dispatch path through a real `CudaProvider`, verify Device residency/no
  Reference CPU fallback, and compare results against the Reference CPU
  Provider; GQA-shaped RoPE with distinct Q/K head counts is also covered
- `magnetar chat` executing every turn of a chat session through one
  persistent Runtime `InferenceSessionId`, with cancellation and close acting
  on that same session
- Quality gates documented in [docs/quality.md](docs/quality.md)

Implemented only as a baseline, fixture, or incomplete production surface:

- caller-facing first-native Qwen execution beyond the built-in `qwen-test`
  model reference
- arbitrary production model artifact parsing and loading from general
  `config.json`, Safetensors shards/indexes, and tokenizer assets
- production model source/hub download flows
- production tokenizer integration beyond deterministic/baseline fixtures
- production continuous batching, prefix-cache reuse, adapters, quantization,
  and multi-device inference
- complete Component host adapters and end-to-end WIT host-call coverage for
  every intended production capability
- `magnetar run`, `magnetar chat`, `magnetar model ...`, `magnetar providers`,
  `magnetar devices`, and `magnetar serve` as fully stabilized production
  service interfaces

Deferred or unsupported for v0.1:

- ROCm, Metal, OpenVINO, QNN, Vulkan, and WebGPU Providers
- production server/API transport
- general model hub downloads
- agent and tool execution inside the Runtime
- concrete Component distribution protocol
- stable Provider ABI

### Important integration boundary

The existence of a working Provider-backed Qwen/CUDA path does **not** mean
Magnetar can already load every arbitrary production Qwen checkpoint supplied
by another application.

Integrators may use the real Runtime, Provider registry, Device discovery, and
Capability/affinity contracts today, but they should fail closed for model
formats or artifact-loading paths Magnetar does not yet support. Missing model
loading or inference functionality must be implemented in Magnetar rather than
recreated in the integrating application.

## Magnetar and Tachyon

Magnetar owns local AI execution. Tachyon, when used, owns distributed service
orchestration: cluster membership, routing, deployment, GitOps, node selection,
and transport-level concerns.

The dependency direction is:

```text
Tachyon
   |
   v
Magnetar
```

Magnetar must remain usable without Tachyon. Tachyon may distribute
Magnetar-compatible Components and model artifacts, but Magnetar validates
Components, controls Capability linking, manages Tensor resources and
residency, selects/uses Providers and Devices, and performs local execution.

Tachyon should consume Provider/Device identities and capabilities reported by
Magnetar rather than fabricating parallel `TensorId`, `PreparedKernelId`,
CUDA-device, memory-capacity, or dtype-support models of its own. If a local
inference capability needed by Tachyon is missing, the capability belongs in
Magnetar.

## Terminology

`Backend`, `Plugin`, and `Host` are not primary Magnetar architectural concepts.
Use Provider for trusted native implementations and Component for portable WASM
extensions. Historical OpenSpec archives may retain older terminology, but
current specifications and architecture documents take precedence.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets
```

The Rust toolchain is pinned in `rust-toolchain.toml` and installed
automatically by rustup. The full set of quality gates, including dependency
and coverage checks, is documented in [docs/quality.md](docs/quality.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change, and
[SECURITY.md](SECURITY.md) for the threat model and known gaps.

APIs are unstable until the first stable release.

## License

MIT. See [LICENSE](LICENSE).
