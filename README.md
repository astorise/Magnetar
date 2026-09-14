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
- Caller-supplied generation parameters and stop conditions on the production
  entry point: `run_production_qwen_generation_for_provider_with_request`
  accepts a `ProductionGenerationRequest` carrying real `GenerationParameters`
  (temperature, top_p, top_k, min_p, typical_p, penalties, seed, banned/
  allowed tokens) and `StopConditions` (token-id and text stop sequences,
  the latter prepared against the real tokenizer internally) and forwards
  them to the same sampling/stop-matching contract every other Runtime
  generation path already uses, plus an optional `max_new_tokens` override
  of the checkpoint manifest's own token budget. Every existing entry point
  (`run_production_qwen_generation`, `_for_provider`, `_for_provider_with_prompt`)
  is now a thin wrapper over this one, reproducing its exact prior greedy/
  default-stop/manifest-token-budget behavior, so no existing caller's
  behavior changes (`expose-production-generation-parameters`)
- Incremental production generation streaming:
  `run_production_qwen_generation_for_provider_streaming` delivers each
  produced token to a caller-supplied callback as it happens (an ordered
  `GenerationStreamEvent::Token { token_id, text_delta }` per token,
  followed by exactly one `Finished { finish_reason, usage }`), instead of
  only returning a final result after decode completes -- real
  time-to-first-token, not the entire generation chopped up after the fact.
  The callback can request clean cancellation
  (`std::ops::ControlFlow::Break`), which stops decode immediately and
  still runs the same session-close/instance-unload cleanup any other
  completion does. Text deltas are produced by re-decoding the accumulated
  token sequence each step and diffing against the previous step's text
  (not `Tokenizer::streaming_decode`/`StreamingDecodeState` -- see the next
  bullet), verified to reconstruct the non-streaming path's decoded text
  exactly (`stream-production-generation-events`)
- **Known gap surfaced while building the above**: `HuggingFaceTokenizer::decode`
  (`loaders/huggingface`, the tokenizer that actually backs every real
  production checkpoint) ignores `DecodeInput.streaming_state` entirely --
  it always returns `pending_partial_state: None` and decodes exactly the
  token slice it is given, regardless of `StreamingDecodeState`. Only
  `FixtureTokenizer`'s own non-delegating byte-fallback path genuinely
  implements the `Tokenizer` Contract's incremental-decode state. This
  does not affect the streaming entry point above (which does not rely on
  `streaming_state`), but means `Tokenizer::streaming_decode` itself is
  not a working incremental-decode primitive for a real production
  tokenizer today -- a real, separate gap for a future `loaders/huggingface`
  change, not fixed here
- A real, external, pinned GGUF file ingestor (`loaders/gguf`), for
  `general.architecture: "qwen2"` and unquantized `F32`/`F16`/`BF16`
  tensors: real architecture-config extraction from GGUF key-value
  metadata, tensor renaming into the same canonical names
  `loaders/huggingface` produces, the same projection-weight transpose and
  tied-`lm_head` derivation `loaders/huggingface` implements for the
  equivalent Safetensors case, and a real byte-level-BPE tokenizer built
  from the GGUF file's own embedded vocabulary -- verified numerically
  identical to `loaders/huggingface`'s output for the same real weight
  values, and against the real public `Qwen2.5-0.5B-Instruct-GGUF`
  checkpoint (`wire-gguf-into-model-loading`)
- Real dequantization of GGUF's `Q8_0`/`Q4_K`/`Q5_K` block-quantized
  tensors to `F32` (ported bit-for-bit from `ggml-org/llama.cpp`'s real
  dequantization source, including the historically error-prone K-quant
  sub-block scale/min packing), exactly mirroring how `F16`/`BF16` storage
  is already converted explicitly: real at Model Loading time
  (`support-gguf-quantized-tensor-dequantization`) *and* reachable from a
  real quantized GGUF file end to end
  (`resolve-gguf-quantized-projection-transpose-sequencing`) --
  `loaders/gguf` dequantizes a quantized tensor before its own
  projection-weight transpose runs (that transpose assumes a flat
  per-element byte width, meaningless for block-quantized data, so
  sequencing dequantize-then-transpose is what makes this correct rather
  than silently corrupting the tensor). Verified against the real public
  `Qwen2.5-0.5B-Instruct-GGUF` checkpoint's actual `Q8_0` export: it loads
  and generates the exact same output as its unquantized F16 GGUF export
  and Safetensors counterpart, for the same prompt. GPTQ/AWQ/BitsAndBytes
  (Hugging Face/Safetensors-shaped quantization schemes, unrelated to
  GGUF's block format) remain entirely unaddressed
- Real native `F16`/`bfloat16` compute on `providers/cuda`, for elementwise
  `add`/`mul`: genuine 2-byte device-resident half-precision buffers, not
  `f32` promoted and labeled, computed via a real on-device kernel and
  verified against an exact reference conversion model (not a tolerance
  band) on real RTX 3070 Ti hardware. Selectable and dispatchable through
  the same Kernel Registry/dispatch contract every other Kernel uses -- a
  `Float16`/`BrainFloat16` `KernelSelectionRequest` selects it in
  preference to the `f32`-only `add`/`mul`, not merely a directly-callable
  Provider primitive anymore (`add-native-cuda-half-precision-compute`,
  `enable-native-cuda-half-precision-elementwise-compute`,
  `wire-cuda-half-precision-into-kernel-registry-dispatch`). Still not
  reachable from any production graph -- nothing requests `Float16`/
  `BrainFloat16` compute for a real generation step -- a half-precision
  resource is not device-resident between separate Kernel invocations the
  way `f32` resources are, and this is not extended beyond `add`/`mul` to
  `matmul`/`rmsnorm`/`rope`/`attention`

Explicitly not yet supported by this profile:

- GPTQ/AWQ/BitsAndBytes quantization (see above -- distinct from and
  unaddressed by GGUF's now-real `Q8_0`/`Q4_K`/`Q5_K` support)
- Native CUDA `F16`/`bfloat16` compute exists for elementwise `add`/`mul`
  only (see above, now genuinely dispatchable through the Kernel Registry,
  but not device-resident between calls and not requested by any
  production graph); `matmul`/`rmsnorm`/`rope`/`attention` still promote
  `F16`/`BF16` weight storage to `F32` at Model Loading time and compute
  there
- LoRA adapters and non-Qwen architecture families
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
loading" above). A follow-up Tachyon-authored review of the OpenAI-compatible
integration surface (not the original charter) found two further P1 gaps in
the production generation entry point specifically: it accepted no caller-
supplied generation parameters or stop conditions, and returned only a final
result rather than incremental streaming events.
`expose-production-generation-parameters` closed the first and
`stream-production-generation-events` closed the second (see "Qwen
production loading" above) -- both P1s from that follow-up review are now
closed. Of the original charter's future-work items, two have since been
substantially addressed: `wire-gguf-into-model-loading` wired
`formats/gguf` into Model Loading for real, `support-gguf-quantized-
tensor-dequantization` made GGUF's `Q8_0`/`Q4_K`/`Q5_K` quantization real
at the Model Loading layer, and `resolve-gguf-quantized-projection-
transpose-sequencing` closed the remaining loader-side gap -- a real,
genuinely quantized public GGUF checkpoint now loads and generates end to
end (see above). GPTQ/AWQ/BitsAndBytes (a different quantization family
entirely, unrelated to GGUF) remain unaddressed, so "quantization
support" as the charter names it broadly is still partial, even though
GGUF's own quantization is now real. Native CUDA `F16`/`BF16` compute has
also progressed through three phases: `add-native-cuda-half-precision-
compute` landed the conversion primitives,
`enable-native-cuda-half-precision-elementwise-compute` landed a real,
hardware-verified on-device `F16`/`bfloat16` `add`/`mul` kernel pair, and
`wire-cuda-half-precision-into-kernel-registry-dispatch` made that pair
selectable and dispatchable through the real Kernel Registry contract --
but this remains a narrow slice, not the charter's full "native CUDA
F16/BF16 compute" item: `matmul`/`rmsnorm`/`rope`/`attention` still
compute in `F32` only, a half-precision resource does not stay
device-resident between separate Kernel invocations the way `f32`
resources do, and no production graph requests half-precision compute for
any real generation step. Everything else the original charter frames as
future work remains exactly that: multi-device execution,
GPTQ/AWQ/BitsAndBytes quantization, and additional Providers
(Metal/ROCm/NPU/TPU) or Model Components (Llama/Mistral/Gemma).

**Tachyon integration audit** (`docs/audits/audit-magnetar-integration-
tachyon-2026-09-13.md`, a separate review from the scope-charter
reconciliation above, of the `inference-components` crate a
`codex/tachyon-component-boundary` branch introduced): found that crate's
generically-named `LoadedInferenceComponent` facade was Qwen/HuggingFace-
specific in behavior despite its generic name (MAG-01), that the
Component WASM artifact it received was registered but never the real
execution authority for generation (MAG-02), and that Component trust was
conflated with Model Artifact trust (MAG-03). MAG-04 (the branch had
diverged from `main`) closed via reconciliation (merge commit `3f101ae`).
`wire-generic-inference-component-runtime` then landed a real, digest-
keyed Component registry in `magnetar-runtime`
(`register_inference_component_artifact`), and
`wire-inference-component-to-generic-registry` wired
`inference-components` to it: `ArtifactTrustPolicy` now separates
Component trust from Model Artifact trust for real (closing MAG-03), and
a caller-registered Component genuinely drives generation end to end via
`ProductionQwenLoadedModel::load_with_component` (closing MAG-02),
verified by a test asserting identical generated tokens between the new
path and the pre-existing hardcoded-singleton path.
`enable-pluggable-model-ingestion-in-inference-components` then closed
MAG-01: `inference-components` selects between `GgufIngestor` and
`HuggingFaceIngestor` (both already implementing the same format-neutral
`ProductionModelArtifactIngestor` contract) by inspecting the bundle it
was given, instead of hardcoding Hugging Face as the only reachable
format -- chat-template rendering needed no format-specific code at all,
since `loaders/gguf`'s own chat-template loader already returns raw text
for the same generic formatter Hugging Face's path already used.
`document-inference-component-concurrency-model` then closed MAG-06:
investigating first (rather than picking a policy from scratch) found
`ProductionQwenLoadedModel::generate`/`generate_streaming` already take
`&mut self`, so one generation in flight per resident instance was
already compiler-enforced, not an open design question -- this change
states that explicitly at `LoadedInferenceComponent`'s own API surface,
changing zero non-comment lines.
`add-second-component-fixture-for-registry-multiplicity-proof` then
closed MAG-07: a second real Component (`synthetic-minimal.component.wasm`,
built through the same real toolchain as the production Qwen Component
but with a deliberately degenerate, decoder-layer-free graph) is
registered alongside the real Qwen Component in the same test, proving
the registry serves two structurally distinct Components at once --
different graphs, neither disturbing the other -- rather than merely
accepting a caller-supplied digest that always resolves back to one
hardcoded singleton. MAG-05 (CI on the consumed commit) is structurally
already satisfied -- every commit pushed to `main`, including this one,
runs the full Quality CI matrix, verified green before and after every
chantier in this list -- what remains outside this repository's scope is
Tachyon actually consuming a post-reconciliation `main` commit, which is
an action on Tachyon's side, not a further Magnetar fix. The Tachyon
audit's full scope (MAG-01 through MAG-07) is closed.
`add-inference-components-integration-test-and-ci-coverage` then closed
the one remaining note left on record: a real, completely-specified
Hugging Face-shaped bundle now drives `LoadedInferenceComponent::load`
through a real generation end to end, proving this crate's own
orchestration (format selection, tokenizer construction, Component
registration, trust evaluation, `load_with_component` wiring) rather than
only the `magnetar-runtime` layer underneath it. Writing that test
surfaced a separate, more serious finding: `inference-components` is its
own `[workspace]` (like `magnetar-cli`, moved out of the root workspace
for the same submodule-dependency reason) but, unlike `magnetar-cli`, was
wired into no CI job at all -- the crate the audit reviewed most closely
had never been built or tested in CI. Fixed by adding it to
`submodule-integration`, with the same format/clippy steps `magnetar-cli`
already has, verified green on the real Linux runner. Outstanding, noted
for the record: a GGUF-shaped equivalent of this end-to-end test (the
`is_gguf` branch itself remains covered separately by `loaders/gguf`'s
own tests and the `magnetar-runtime` singleton/named-component tests).

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
