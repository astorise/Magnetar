# v0.1 Release Statement

Magnetar v0.1 is a CPU-local inference runtime baseline.

Included baseline:

- Runtime-owned inference orchestration through a Runtime-registered generation
  executor.
- Reference CPU baseline contracts and local conformance coverage.
- Runtime session, generation, sampling, tokenizer, memory, model-loading, and
  trust-boundary contracts.
- Supported WIT packages:
  - `magnetar:compute@2.0.0`
  - `magnetar:observability@1.0.0`

Preview:

- CLI command surface and local boundary harnesses.
- Model, tokenizer, and adapter artifact metadata integration.

Deferred or unsupported:

- Full KernelDispatch/ProviderExecution-backed E2E logits.
- Incremental decode/KV-cache execution as the required generation path.
- CUDA, Metal, OpenVINO, QNN, WebGPU.
- Production server API.
- Model hub downloads.
- Agent/tool Runtime.
- SBOM generation and artifact signing.

No release claim is made for raw prompt storage, raw tensor reporting, raw KV
cache reporting, raw native handles, arbitrary filesystem access, process
execution, network access, or secret access through the Runtime.
