# Magnetar

Magnetar is a Rust runtime for portable local AI execution.

The current implementation is a v0.1 local-runtime baseline. The repository
contains the `magnetar-runtime` and `magnetar-cli` crates, executable contract
tests, and the OpenSpec history that defines the architecture as it grows.
Provider-backed full model execution is still incomplete; the baseline focuses
on CPU-local contracts, fixtures, and fail-closed boundaries.

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
- Memory planning and TensorResource contracts
- Operator, Kernel, Kernel Registry, Kernel Dispatch, and Reference CPU
  Provider contracts
- Model Artifact, Model Loading, Model Instance, Tokenizer, Generation,
  Sampling, Session, KV Cache, Prefix Cache, Continuous Batching, Runtime
  Inference API, and E2E conformance contract surfaces
- Fixture-backed local inference and conformance paths used to validate the
  runtime boundaries
- Quality gates documented in [docs/quality.md](docs/quality.md)

Implemented as baseline fixture or contract-only:

- complete Component host adapters and end-to-end WIT host-call fixtures
- Runtime-owned Provider-backed generation through graph validation, Kernel
  Registry, Kernel Dispatch, TensorResource logits, and Sampling
- incremental prefill/decode execution using KV cache
- production model artifact parsing, residency, and hub/source downloads
- production tokenizer integration beyond deterministic fixtures
- production continuous batching, prefix cache reuse, adapters, quantization,
  and multi-device inference
- `magnetar run`, `magnetar chat`, `magnetar model ...`, `magnetar providers`,
  `magnetar devices`, and `magnetar serve` as CLI boundary harnesses rather
  than production service commands

Deferred or unsupported for v0.1:

- CUDA, ROCm, Metal, OpenVINO, QNN, Vulkan, and WebGPU Providers
- production server/API transport
- model hub downloads
- agent and tool execution inside the Runtime
- concrete Component distribution protocol
- concrete Provider ABI stabilization

## Magnetar and Tachyon

Magnetar is intended to own local AI execution. Tachyon, when used, owns
distributed service orchestration: cluster membership, routing, deployment,
GitOps, and node selection.

The dependency direction is:

```text
Tachyon
   |
   v
Magnetar
```

Magnetar must remain usable without Tachyon. Tachyon may distribute
Magnetar-compatible Components and model artifacts, but Magnetar validates
Components, controls Capability linking, and performs local execution.

## Terminology

`Backend`, `Plugin`, and `Host` are not primary Magnetar architectural concepts.
Use Provider for trusted native implementations and Component for portable WASM
extensions. Historical OpenSpec archives may retain older terminology, but
current specifications and architecture documents take precedence.

## Development

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

APIs are unstable until the first stable release.
