# Magnetar

Magnetar is a Rust runtime for portable local AI execution.

The current implementation is a v0.1 local-runtime baseline with real
Provider-backed first-native execution. The repository contains the
`magnetar-runtime` and `magnetar-cli` crates, executable contract and
integration tests, and the OpenSpec history that defines the architecture as
it grows.

Reference CPU execution is implemented, and an external CUDA Provider is now
integrated and validated through real-hardware first-native end-to-end tests.
Production Qwen model artifact loading -- real `config.json` +
Safetensors/tokenizer inputs, not only the built-in `qwen-test` path -- is
implemented and verified end to end, including against a real, publicly
downloaded Qwen2.5-0.5B-Instruct checkpoint on both Reference CPU and real
CUDA hardware. Magnetar is **still not a general production model runtime**:
see the "Qwen production loading" subsection below for the precise profile
this covers and what it does not yet.

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
- Production Qwen model artifact ingestion: an external, pinned ingestor
  (`loaders/huggingface`) parses real Hugging Face-style `config.json`,
  `tokenizer.json`/`tokenizer_config.json`, and single-file or sharded
  Safetensors bundles into the same generic `ModelManifest`/`ModelTensorMetadata`
  contracts Model Loading already used, with no concrete format dependency
  inside `magnetar-runtime` itself. `magnetar model load --file <path>` uses
  this path for local bundles instead of a fixture manifest.
- Caller-facing first-native Qwen generation for a production-ingested
  `ModelInstance`, not only the built-in `qwen-test` model reference, through
  the same real compiled Qwen Component graph production and generic Runtime
  Inference API every other first-native path uses. Verified end to end
  against real ingested bundles on both Reference CPU and real CUDA hardware
  (RTX 3070 Ti Laptop GPU); see the "Qwen production loading" subsection below
  for the profile this currently covers and what it does not yet.
- A real `tokenizer.json`-backed tokenizer implementation (via the
  `tokenizers` crate, kept entirely inside the external ingestor module, never
  a `magnetar-runtime` dependency), behind the same Tokenizer Contract the
  deterministic fixture tokenizer implements
- Quality gates documented in [docs/quality.md](docs/quality.md)

Implemented only as a baseline, fixture, or incomplete production surface:

- production model source/hub download flows (a caller supplies an
  already-authorized local bundle; Magnetar does not fetch one itself)
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

### Qwen production loading

See
[docs/production-model-loading-integration.md](docs/production-model-loading-integration.md)
for the minimal embedder integration recipe (Tachyon-Mesh or otherwise).

The first production Qwen profile currently supports, verified end to end
against real ingested bundle bytes on both Reference CPU and real CUDA
hardware:

- Single Qwen decoder architecture family (Hugging Face `Qwen2ForCausalLM`
  config shape), any layer/head/hidden/intermediate-size combination,
  including grouped-query (distinct attention/KV head counts) configurations
- Real `config.json` parsing and validation (rejects missing/zero/
  inconsistent architecture values structurally)
- Single-file and Hugging Face-style indexed sharded Safetensors, `F32`/`F16`/
  `BF16` storage (decoded and converted to `F32` for compute), real
  per-tensor Safetensors parsing (`formats/safetensors`, no second parser)
- Real `tokenizer.json` (via the `tokenizers` crate) plus
  `tokenizer_config.json`/`generation_config.json` normalization, including a
  tokenizer vocabulary smaller than the model's declared `vocab_size` (a real,
  common Hugging Face convention: the embedding table is padded to a
  hardware-friendly round number past the tokenizer's actual vocabulary)
- Tied-embedding (`tie_word_embeddings: true`) checkpoints: `lm_head` is
  derived from `token_embedding` at load time, driven by real config/tensor
  metadata (untied checkpoints, which declare their own `lm_head.weight`
  tensor, are unaffected)
- Real Qwen2/2.5 attention bias: `self_attn.{q,k,v}_proj` bias terms (an
  architectural default of the model class, not a `config.json` field) are
  ingested and applied via a broadcast add on both Reference CPU and CUDA;
  `o_proj` and every MLP/`lm_head` projection never carry one in this baseline
- An authorized local or Tachyon-sourced bundle (a caller-supplied,
  already-authorized source; Magnetar does not fetch or discover one itself)
- A real, publicly downloaded Qwen checkpoint
  ([`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
  pinned by revision and content digest), not only tiny synthetic-content
  bundles: verified loading and real decoded-text generation on Reference CPU,
  and a matching greedy 16-token decode (text and every generated token id)
  between Reference CPU and real CUDA hardware within an exact-match
  tolerance (see `integration-tests/production-loading`'s
  `tests_real_checkpoint_smoke.rs`, a manual/nightly-profile test given its
  ~1GB download)
- Real device-resident multi-step CUDA decode: a generation request needing
  more than one decode step now succeeds on CUDA, not just prefill.
  Historical KV concatenation across decode steps dispatches through a real,
  device-resident `concat` Kernel (a live Kernel Registry selection, since
  this Runtime-side bookkeeping is not itself a graph node with a Prepared
  Plan binding) instead of downloading the historical KV tensor to host, and
  the KV pending-write/commit lifecycle uses a new
  `ProviderExecutionApi::copy_tensor_admitted` primitive (a real
  device-to-device copy) instead of a download-then-reupload round trip --
  verified byte-for-byte identical against Reference CPU on both a synthetic
  bundle (8 tokens) and the real public Qwen2.5-0.5B-Instruct checkpoint (16
  tokens), on real CUDA hardware
  (`implement-device-resident-multi-step-cuda-decode`)
- A real, artifact-declared chat template: when the ingested bundle's
  `tokenizer_config.json` declares one, `loaders/huggingface`'s
  `HuggingFaceChatTemplateFormatter` renders `PromptInput::ChatMessages`
  through the real Jinja2 template (verified byte-exact against a real
  Qwen2.5-Instruct template, and separately against a real Phi-3-mini
  template's materially different markup) instead of a plain-text
  placeholder -- verified on the real public checkpoint: the same real
  weights produce the incoherent `" 1000000"` for an unformatted plain-text
  prompt but a coherent `"The capital of France, France."` once rendered
  through the checkpoint's own real template for the same question
- An explicit, checked-early `InferenceApiError::Unsupported` when a
  generation request needs more than one decode step against a Provider
  that declares (`Provider::supports_multi_step_decode`) it cannot supply
  host-readable KV history for it -- checked before any real execution
  work, not discovered as an internal residency error partway through a
  real prefill. CUDA itself now declares support (see above); this remains
  a real, generic mechanism for a future Provider that genuinely cannot
  support the shape
- Real measured `tokens_per_second`/`prefill_duration_millis`/
  `decode_duration_millis` generation usage metadata, for both Reference
  CPU and real CUDA hardware, from actual wall-clock timing (verified on
  both) rather than left unset

Explicitly not yet supported by this profile:

- `F16`/`BF16` weight storage is decoded and converted to `F32` at Model
  Loading time, but native CUDA `F16`/`BF16` compute kernels do not exist yet
- GGUF, quantized (`Q4_K`/`Q5_K`/`Q8_0`/GPTQ/AWQ/BitsAndBytes) execution, LoRA
  adapters, and non-Qwen architecture families
- Remote model hub download, OCI distribution, and credential/retry handling
- Wiring `magnetar-cli` to run generation against a production-loaded
  instance at all: `magnetar model load --file` proves real ingestion and
  loading, but no CLI command then generates from that instance -- `magnetar
  run`/`magnetar chat` remain bound to the separate `qwen-test` Component
  fixture. A real, separate CLI feature, not attempted by
  `close-tachyon-scope-audit-gaps` (see that change's proposal Non-Goals)

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

**Qwen production loading cutover criterion**: `implement-production-qwen-model-loading`
task group 12 (Reference CPU and CUDA prefill proofs, the real public
checkpoint smoke test, CPU/CUDA output comparison, and release-gate
hardening) is now closed, so Tachyon may remove any fail-closed placeholder
standing in for real Qwen model loading and route through the public
production loading/generation surface (`load_production_qwen_instance`/
`run_production_qwen_generation` and their Provider-generic variants,
`ProductionModelArtifactIngestor`, and the `loaders/huggingface` external
ingestor) instead. Tachyon should keep parsing no model formats and managing no weight
resources itself either way (see "Public embedder / Tachyon loading surface"
in that change's proposal); this criterion is about when its fallback path
becomes safe to delete, not about which side does the parsing.

**Tachyon scope charter reconciliation**: reconciling the shipped code against
a Tachyon-authored Magnetar scope charter found the large majority of it
already real and verified. `close-tachyon-scope-audit-gaps` closed two
concrete gaps that reconciliation found (real chat template rendering; an
explicit `Unsupported` signal for a decode shape a Provider cannot perform,
plus real `tokens_per_second` measurement), and
`implement-device-resident-multi-step-cuda-decode` closed the third
(device-resident multi-step CUDA decode itself, see "Qwen production
loading" above). Everything else the charter frames as future work remains
exactly that: multi-device execution, quantization support, wiring
`formats/gguf` into Model Loading, native CUDA `F16`/`BF16` compute, and
additional Providers (Metal/ROCm/NPU/TPU) or Model Components
(Llama/Mistral/Gemma).

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
