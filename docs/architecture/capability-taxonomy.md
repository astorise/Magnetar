# Capability contract taxonomy

## Status and scope

This document records the evidence used to prepare future Magnetar capability
contracts. It is a discovery artifact, not a registry of stable capabilities:
the package names below are provisional, and this change does not alter
`magnetar:compute/run@1.0.0` or add runtime APIs.

The taxonomy distinguishes three things that source runtimes often combine:

- native responsibilities owned by a Provider, Device, or runtime service;
- WIT-backed Capability candidates consumable by portable Components;
- hybrid candidates whose WIT surface controls Provider-owned opaque resources.

A native responsibility is not a Magnetar Capability. Existing Magnetar
Capabilities remain versioned contracts with at least one WIT interface.

## Reproducible evidence

| Source | Pinned revision | Reviewed scope |
| --- | --- | --- |
| [Hugging Face Candle](https://github.com/huggingface/candle/tree/2a13b0f3ff62f7e67013597f2996f764c5735e21) | `2a13b0f3ff62f7e67013597f2996f764c5735e21` (`0.11.0`, 2026-08-14) | `candle-core` device, backend, storage, tensor, shape/layout/dtype, operation, and module surfaces |
| [Crane](https://github.com/lucasjinreal/Crane/tree/a47b11ce9d36f269d3c100e1f84716b3dbf23777) | `a47b11ce9d36f269d3c100e1f84716b3dbf23777` (2026-08-18) | `crane-core` generation/tokenization and `crane` application abilities |

Neither runtime is a Magnetar dependency or submodule. The revisions are
recorded solely to make the analysis repeatable.

### Review method

1. Inspect public data types, traits, and their required methods at the pinned
   revision.
2. Inspect representative implementations and call sites when a trait alone
   does not reveal state, output, or lifecycle semantics.
3. Group responsibilities by semantic cohesion, resource lifetime, versioning
   pressure, and fallback behavior; do not mirror a Rust module or trait merely
   because it exists.
4. Map each group to Magnetar's existing Provider, Device, Capability,
   Component, or runtime-service roles.
5. Propose a WIT boundary only when its values and lifecycle can be expressed
   portably. Keep raw storage, hardware handles, Rust generics, callbacks, and
   concrete models out of that boundary.

Source references use immutable GitHub `blob/<commit>/<path>` links followed by
the reviewed Rust symbol. Line numbers are intentionally omitted because the
symbol is stable within the pinned blob and remains searchable.

## Classification vocabulary

| Disposition | Meaning |
| --- | --- |
| Native-only | Provider, Device, or runtime implementation detail; not a Capability candidate |
| Component-suitable | Serializable semantic contract suitable for direct WIT import/export |
| Hybrid | WIT control surface over Provider-owned opaque resources or streams |

Fallback labels used later in the document are:

| Fallback | Meaning |
| --- | --- |
| Transparent | Another compatible Provider can be selected before observable work without recreating state |
| Restartable | Complete input can be replayed on another Provider, but the operation or session must restart |
| Provider-pinned | Live resources or observable output prevent transparent switching; explicit teardown or failure is required |

## Candle findings

### Device, backend, storage, and module boundaries

| Source evidence | Observed responsibility | Magnetar mapping | Contract consequence |
| --- | --- | --- | --- |
| [`DeviceLocation`, `Device`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/device.rs) | Identifies CPU/CUDA/Metal locations, owns concrete backend handles, constructs devices, compares locations, selects a backend, seeds random generation, allocates storage, and synchronizes work. Multiple `Device` values may share one physical `DeviceLocation`. | Physical identity and descriptive information map to Magnetar `Device`; discovery and concrete handles remain Provider-owned. | Native-only. Do not copy Candle's closed backend enum or direct constructors into WIT. Components select requirements/capabilities, not CUDA or Metal handles. |
| [`BackendDevice`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/backend.rs) | Associates a device with storage; creates it from an ordinal; allocates initialized/uninitialized buffers; uploads CPU storage; generates random values; owns seed state; and synchronizes queued work. | Provider implementation plus runtime memory planner/scheduler. The existing Magnetar `Device` is metadata and identity, not this execution object. | Native-only implementation surface. A future WIT capability may request allocation or execution through opaque resources, but cannot expose `unsafe` allocation, associated Rust types, or backend storage. |
| [`BackendStorage`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/backend.rs), [`Storage`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/storage.rs) | Owns dtype/device-specific memory and dispatches clone/download, conversion, elementwise, reduction, comparison, convolution, pooling, indexing, copy, matrix multiplication, and mutation kernels against explicit layouts. | Provider-owned buffer and kernel implementation behind the Compute capability and future memory/scheduling services. | Native-only representation and dispatch. A WIT interface may submit coarse work over opaque buffers; it must not reproduce `BackendStorage`, mutable borrows, layout references, or per-backend storage enums. |
| [`Tensor`, `Tensor_`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/tensor.rs), [`Shape`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/shape.rs), [`Layout`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/layout.rs), [`DType`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/dtype.rs) | Couples a reference-counted storage, shape/strides, dtype, device, mutation lock, and backprop metadata; validates high-level operations before dispatching to storage. Views can share storage. | Portable descriptor values plus a Provider-owned opaque tensor/buffer resource. Graph validation belongs to a future Compute frontend or compiler service. | Hybrid evidence. Shape/dtype and operation descriptions can cross WIT; storage, locks, device handles, and graph internals remain native. Magnetar does not adopt Candle's eager Tensor API. |
| [`Module`, `ModuleT`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/lib.rs) | Defines a single-tensor forward transform; `ModuleT` adds a training/evaluation flag and automatically wraps `Module`. | Native model/operator implementation invoked by a future graph or model execution contract. | Native-only as written. A universal WIT model contract cannot assume one tensor input/output, and training behavior is outside Magnetar's inference scope. |

The important separation is that Candle's `Device` is both a public selector and
an executable backend handle, while Magnetar already separates immutable
`Device` metadata from the native Provider that owns execution. The taxonomy
preserves Magnetar's separation.

### Tensor operation families

[`Tensor`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/tensor.rs)
and [`Op`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/op.rs)
provide evidence for the following semantic families. These are coverage areas
inside future Compute revisions, not automatically one Capability per row.

| Operation family | Representative source surface | Ownership and exclusions |
| --- | --- | --- |
| Descriptor and views | shape/dims, dtype, layout, reshape, flatten, squeeze/unsqueeze, transpose/permute, narrow/slice, broadcast | Descriptors may be WIT values; aliases, strides, and shared storage remain Provider-owned. |
| Construction and allocation | scalar/slice construction, `zeros`, `ones`, ranges, uninitialized allocation | Allocation policy, memory pools, and unsafe initialization remain native; portable requests use validated descriptors. |
| Data movement and conversion | `to_device`, `to_dtype`, contiguous copy, host upload/download | Runtime planner and Providers own placement and transfer. A future coarse transfer operation must make copies and resource ownership explicit. |
| Elementwise and selection | unary/binary arithmetic and activations, comparisons, affine/power, `where_cond` | Candidate Compute graph operations; Rust operator traits and scalar generics are excluded. |
| Reductions | sum/mean/min/max/argmin/argmax across selected dimensions | Candidate Compute graph operations with explicit axis, keep-dimension, dtype, and empty-input semantics still to specify. |
| Linear algebra | matrix multiplication and broadcast matrix multiplication | Candidate Compute graph operations; batching, transpose, accumulation dtype, quantization, and precision policy remain unresolved. |
| Convolution and spatial transforms | convolution/transposed convolution, pooling, nearest/bilinear upsampling | Candidate Compute graph operations; layout, padding, dilation, and numerical semantics require a dedicated contract revision. |
| Indexing and updates | gather, index-select/add, scatter/set/add, slicing and concatenation | Candidate Compute graph operations. Mutation and aliasing must be replaced by explicit result/resource semantics at WIT boundaries. |
| Random generation | uniform/normal generation plus device seed state | Hybrid at most: portable distribution/seed policy over Provider-owned random state. Determinism across Providers is not assumed. |
| Synchronization | device-wide synchronization after queued work | Native scheduler/Provider primitive. Components should await coarse operations or streams, not synchronize a hardware queue directly. |
| Autograd, variables, and custom Rust operations | backprop metadata, variables, `CustomOp*`, in-place Rust traits | Excluded: training is a project non-goal, and Rust trait objects are not portable contracts. Future custom operations enter through a separately specified Component/Provider extension boundary. |

This grouping deliberately keeps hot per-operation dispatch inside a Provider.
A future Compute contract should favor graph/batch submission or another coarse
interface so WIT crossings do not become the inner tensor loop.

## Crane findings

### Model and generation-session boundary

[`ModelForCausalLM`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-core/src/generation/based.rs)
combines a native Candle `Device`, mutable model state, a complete generation
loop, and an optional Rust streamer. Its default `generate` implementation
returns generated token IDs only, while the reviewed Qwen, Gemma, Hunyuan, and
MiniCPM overrides return `prompt IDs + generated IDs`. The higher-level
[`LlmClient`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/llm/client.rs)
assumes the latter and slices off the prompt. EOS choice and streamer
finalization also vary by model.

The mutable receiver used by concrete models hides KV-cache and tokenizer
state inside the model object. Magnetar therefore needs two boundaries instead
of this trait:

- a native model backend owns weights, device placement, KV caches, sampling,
  batching, and forward steps;
- a hybrid generation Capability creates an opaque session and returns
  structured events and a terminal result.

Crane's server makes this separation more explicit in
[`ModelBackend`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-serve/src/engine/backend.rs),
[`EngineRequest`, `GenerationParams`, and `EngineResponse`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-serve/src/engine/types.rs),
and [`Sequence`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-serve/src/engine/sequence.rs).
`ModelBackend` remains native because it exposes Candle tensors, devices, and
cache operations; the request/session/event facade is the portable candidate.

### Generation policy

[`GenerationConfig`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-core/src/generation/mod.rs)
is duplicated in the application crate and mixes distinct concerns. The
reviewed causal generation loops do not consume `do_sample` or `pad_token_id`,
and multiple implementations hide a fixed random seed. Magnetar must specify
the behavior rather than inherit the field names.

| Crane concern | Magnetar disposition |
| --- | --- |
| `max_new_tokens` | Portable termination limit in a generation request. |
| `temperature`, `top_p`, repetition penalty/window | Portable sampling policy only after value ranges, greedy behavior, ordering, and validation are specified. |
| `do_sample` | Do not copy; encode the sampling strategy explicitly so it cannot disagree with temperature/top-p. |
| `pad_token_id` | Model/tokenizer binding metadata, not a universal per-request control. |
| singular `eos_token_id` | Replace with model defaults plus optional `stop-token-ids: list<u32>` and text stop conditions. |
| hidden fixed seed | Add an optional explicit seed and document Provider-level determinism; absence means Provider-selected randomness. |
| `enable_thinking` | Prompt/chat-template policy, not causal decoding policy. |
| `report_speed` | Telemetry subscription/response metadata, not generation semantics. |

A future result also needs generated output separate from the input prompt,
`finish-reason`, usage counters, and stable structured errors. These remove the
output and finalization ambiguities observed in Crane.

### Token streaming

[`TokenStreamer`, `TextStreamer`, and `AsyncTextStreamer`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-core/src/generation/streamer.rs)
use Rust callbacks and `std::sync::mpsc`; they are native adapters, not a WIT
ABI. Decoding one token at a time is also not a stable text protocol because
BPE and Unicode output may require retained prefix state, which Crane handles
separately in
[`TokenOutputStream`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-core/src/utils/token_output_stream.rs).

The portable shape is a pull-based generation-session resource with
backpressure and cancellation. Its ordered events distinguish token-ID deltas,
incrementally decoded text deltas, usage, completion with a finish reason, and
failure. Events carry sequence numbers so a client can detect restart or
duplication. Streaming is a protocol pattern used by an ability, not an
independent Capability with an untyped payload.

### Tokenizer and prompt formatting

[`AutoTokenizer`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-core/src/autotokenizer.rs)
combines local/Hugging Face/GGUF acquisition, tokenizer configuration,
encoding/decoding, special-token lookup, and Jinja chat-template rendering.
These responsibilities have different trust, caching, and versioning needs:

| Responsibility | Boundary |
| --- | --- |
| Artifact acquisition and cache | Native model/artifact service; Components pass a host-authorized source or digest, never ambient filesystem paths or credentials. |
| Encode/decode and special-token metadata | Component-suitable facade backed by a native or portable implementation. |
| Incremental text decode | Stateful tokenizer/session resource so partial byte sequences remain valid. |
| Chat template, tools, and reasoning options | Separate Component-suitable prompt-formatting contract. |

The loaded model, tokenizer vocabulary, special tokens, and chat template must
share a content fingerprint. A fallback is compatible only when that binding
is preserved; matching a model family name is insufficient.

### Application abilities and source maturity

| Ability evidence | Maturity at the pinned revision | Taxonomy consequence |
| --- | --- | --- |
| [`LlmClient`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/llm/client.rs), [`ChatClient` and chat types](https://github.com/lucasjinreal/Crane/tree/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/chat) | Implemented text generation, chat history, and callback streaming, subject to the generation inconsistencies above. | Evidence for generation and application-level chat orchestration as separate boundaries. |
| [`Asr`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/audio/asr.rs) | Defines complete/incremental transcription, transcript language/finality, input sample rate, and supported languages; concrete model implementations remain native. | Strong evidence for a high-level speech-recognition contract over media resources/chunks. |
| [`Tts`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/audio/tts.rs) | Defines voices, audio format, synthesis, optional voice cloning, and chunks; the generic `TtsClient` is still a placeholder, while model-specific implementations exist. | Evidence for speech synthesis, with voice cloning and streaming advertised as negotiated features rather than universal methods. |
| [`OcrClient`, `OcrDocument`, `OcrRegion`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/vision/ocr.rs) | OCR executes, but region detail is backend/feature-dependent and can be empty for the vision-language path. | Evidence for OCR with optional/negotiated structured regions. |
| [`VisionClient`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/vision/image_analysis.rs), [`MultimodalClient`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/multimodal/vision_language.rs) | Explicit placeholders that return not-implemented errors; server-side VLM handlers are more concrete but not a stable SDK contract. | Vision-language remains a provisional candidate. The placeholder clients are not evidence of stable semantics. |

Advertised duplex speech, VAD, model-specific tools, and other routes are not
promoted into the initial taxonomy without the same source and lifecycle
analysis. Their existence is implementation evidence, not a portability
guarantee.

## Proposed Magnetar taxonomy

Each responsibility has one primary layer below. A row marked native-only is a
runtime role, not a Capability. A Component-suitable or hybrid row is a
candidate only; it still requires a dedicated OpenSpec change and at least one
versioned WIT interface before registration.

### Low-level execution layer

| Responsibility family | Responsibilities | Magnetar role and disposition | Direct dependencies | Explicit exclusions | Evidence |
| --- | --- | --- | --- | --- | --- |
| Device catalog and identity | Provider discovery, globally unique identity, immutable hardware metadata, advertised execution support | Existing `Device` plus Provider registry; native-only service, with portable metadata values | Provider registration | Execution handles, queues, allocation, direct Component hardware selection | Candle `DeviceLocation`; existing Magnetar device spec |
| Execution context | Provider/device affinity, queues or streams, seed state, command submission, synchronization domain | Provider and scheduler; native-only | Device identity | Raw CUDA/Metal handles in WIT; treating physical location as an interchangeable execution context | Candle `Device`, `BackendDevice` |
| Memory and allocation | Allocate/free, memory pools, capacity/alignment, initialized storage, host-visible staging | Provider plus future memory planner; native-only | Execution context, shape, dtype | `unsafe` allocation, pointers, allocators, and backend storage enums in WIT | Candle `BackendDevice`, `Storage` |
| Tensor descriptor and resource | Fixed-width shape/dtype metadata, validated layout/view descriptions, opaque data lifetime, alias/materialize distinction | Shared WIT values plus a hybrid opaque resource inside a future Compute revision | Memory/allocation, execution context | Candle `Tensor`, `Arc`, locks, `usize`, backprop graph, and implicit aliasing semantics | Candle `Tensor`, `Shape`, `Layout`, `DType` |
| Data movement | Explicit upload/download/copy, dtype conversion, placement transfer, staging and synchronization cost | Runtime planner and hybrid Compute operations over opaque resources | Source/destination tensor resources and execution contexts | Implicit CUDA/Metal interop or silent CPU staging | Candle `to_device`, `to_dtype`, storage copy methods |
| Compute graph execution | Validate and submit coarse graphs/batches covering elementwise, reduction, linear algebra, indexing, convolution, spatial, and view operations | Hybrid extension candidate for the existing `magnetar:compute/run` Capability | Tensor resources, allocation, execution context | One WIT call per eager tensor primitive; backend kernel names; training/autograd | Candle `Tensor`, `Op`, `BackendStorage` |
| Random execution | Distribution parameters, explicit optional seed, generation into an opaque tensor | Hybrid interface group within Compute, backed by Provider state | Tensor allocation, execution context | Cross-Provider bitwise determinism or hidden global seed guarantees | Candle device random/seed methods |
| Synchronization and completion | Await submitted coarse work and surface completion/errors to the scheduler | Runtime scheduler and Provider; native-only primitive surfaced indirectly by async operation/session completion | Execution context | Component-visible device-wide queue synchronization | Candle `Device::synchronize` |
| Module/operator execution | Invoke compiled/native forward units and manage their device affinity | Provider-native model/graph implementation | Compute graph execution, tensor resources | Candle's single-tensor signature as a universal model ABI; `ModuleT::train`; lifecycle assumptions | Candle `Module`, `ModuleT` |
| Custom kernels and autograd | Backend-specific custom operations and training graph | Provider extension or future separate change; native-only and out of initial scope | Tensor resources, execution context | Arbitrary Rust trait objects through WIT; training in the inference runtime | Candle `CustomOp*`, `BackpropOp` |

Shape dimensions and strides need fixed-width WIT integers with overflow and
product validation; Candle's platform-sized `usize` is not a portable contract.
Likewise, a tensor view and a materialized copy must be distinguishable because
Candle can share storage for a view but copy when layout constraints require it.

### Model execution layer

| Responsibility family | Responsibilities | Magnetar role and disposition | Direct dependencies | Explicit exclusions | Evidence |
| --- | --- | --- | --- | --- | --- |
| Artifact resolution | Authorize and fetch content-addressed model/tokenizer artifacts, cache them, verify digest/format | Host runtime service; native-only | Host policy and storage | Ambient paths, network credentials, Hugging Face clients, or mmap handles in WIT | Crane `AutoTokenizer::from_pretrained`, model factory |
| Model loading | Load a model from an authorized artifact, select format/dtype constraints, return opaque model metadata and advertised abilities | Hybrid Capability candidate; WIT facade over a Provider-owned model resource | Artifact resolution, Device, memory, Compute | Closed model architecture enum; raw weights; Provider/device choice by a Component | Crane `LlmClient::new`, server model factory |
| Tokenization | Encode/decode, incremental decoding, special-token metadata, tokenizer fingerprint | Component-suitable Capability candidate; native implementation allowed | Tokenizer artifact bound to the model fingerprint | Artifact acquisition, credentials, chat rendering, per-token stateless text decoding | Crane `AutoTokenizer`, `TokenOutputStream` |
| Prompt formatting | Render messages, tools, and reasoning policy to model input using a fingerprinted template | Component-suitable Capability candidate or portable Component export | Tokenization metadata and chat-template artifact | Sampling, model forward pass, arbitrary host template file access | Crane chat-template methods and server processor |
| Generation session | Create/cancel a causal generation session; accept semantic sampling/stopping policy; emit ordered deltas, usage, finish reason, and errors | Hybrid Capability candidate over an opaque Provider-owned session | Loaded model, Compute; tokenizer only for optional text facade | Candle device/tensor, mutable model object, callbacks/channels, prompt-prefixed output convention | Crane `ModelForCausalLM`, server engine request/sequence |
| Generation event protocol | Token/text delta, sequence number, usage, completion/failure; pull/backpressure/cancel semantics | WIT protocol nested in each streaming ability, not a standalone Capability | Owning generation or application session | Untyped generic streamer and Rust callback/channel ABI | Crane `TokenStreamer`, `StreamerMessage` |
| Inference engine internals | Forward steps, KV cache, cache swap, batch decode, hot sampling loop, scheduling | Provider and scheduler; native-only | Loaded model, Compute, memory, execution context | Tensor/device/cache handles in WIT and transparent migration of live state | Crane server `ModelBackend` |

Model and tokenizer resources must carry compatible artifact, vocabulary, and
template fingerprints. A dependency by Capability ID/version alone cannot
prove that two independently resolved resources belong to one coherent model
bundle.

### Application ability layer

| Responsibility family | Responsibilities | Magnetar role and disposition | Direct dependencies | Explicit exclusions | Evidence |
| --- | --- | --- | --- | --- | --- |
| Text completion facade | Text input/output, optional incremental deltas, usage and finish reason | Portable Component export or Component-suitable facade | Tokenization, generation session | Native model/device selection and duplicated sampling engine | Crane `LlmClient` |
| Chat/conversation | Message/history orchestration, roles, tools, prompt policy, streamed assistant response | Portable Component export; may become a WIT-backed application Capability when Component exports participate in resolution | Prompt formatting, tokenization, generation session | Owning weights/KV cache; hard-coded model templates; assuming runtime currently resolves Component-backed Capabilities | Crane `ChatClient`, chat/server API types |
| Speech recognition | PCM/media input, language hints, partial/final transcript chunks, supported-language metadata | Hybrid Capability candidate with Provider-native media/model execution | Media decode/resample, loaded model, Compute; tokenization/generation when model requires them | Filesystem paths, native PCM pointers, universal support for partial results | Crane `Asr`, `TranscribeOptions` |
| Speech synthesis | Text/language/voice request, negotiated voice-clone support, audio metadata/chunks | Hybrid Capability candidate with Provider-native codec/model execution | Loaded model, Compute, audio codec; tokenization/generation when the architecture requires them | Candle tensor waveforms in WIT, ambient reference-audio paths, changing voice/format mid-stream | Crane `Tts`, `SpeechOptions` |
| Optical character recognition | Image input, text result, optional regions/confidence, streaming when supported | Hybrid Capability candidate | Image decode/preprocess, loaded model, Compute; tokenization/generation for VLM backends | Requiring regions from every backend or exposing image file paths | Crane `OcrClient` |
| Vision-language | Image plus structured text/messages, generated multimodal response | Provisional hybrid Capability candidate | Image preprocess, prompt formatting, tokenization, loaded model, generation | Claiming stable semantics from placeholder SDK clients; raw image/tensor copies per token | Crane server VLM path; SDK placeholders only as negative evidence |

Application orchestration can live in a portable Component, while model-heavy
abilities may initially be implemented by native Providers. Both use WIT
interfaces; the current runtime does not yet register a Component export as a
Provider-backed Capability, so that resolution path is a future design item.

### Dependency graph

```text
Provider registration
  -> Device catalog
       -> Execution context
            -> Memory/allocation
                 -> Tensor resource
                      -> Data movement
                      -> Compute graph execution
                           -> Module/inference internals

Host artifact service -> Model loading -> Generation session
  -> tokenizer artifact -> Tokenization
  -> template artifact --> Prompt formatting

Generation session + Tokenization ----------------> Text completion
Generation session + Tokenization + Prompt format -> Chat
Model loading + Compute + media services ---------> ASR / TTS / OCR
Model loading + Compute + image preprocessing
  + Prompt formatting + Tokenization + Generation -> VLM

Optional, model-dependent edges:
  Tokenization / Generation - - -> ASR / TTS / OCR
```

For ASR/TTS/OCR/VLM, media decode, preprocess, resample, and codec nodes are
native services until their own portable semantics are specified. Solid edges
show required dependencies for the responsibility as classified here; dashed
edges are architecture-dependent and must be negotiated rather than assumed.
Following the solid arrows exposes only the universal transitive dependencies.

The current `CapabilityDescriptor.dependencies` model validates global
availability by ID and version. Stateful chains need more: the runtime must
select a coherent set of implementations, bind opaque resources to their
Provider and Device, and keep later calls on that affinity. Resolving every
node independently can produce an unusable model/tokenizer/device combination.

## Contract preparation

### Provisional WIT package map

These names are design inputs only. No entry is registered, versioned, or
implemented by this change. Each row needs its own follow-up proposal, and a
follow-up may rename, split, or reject it.

| Candidate family | Provisional package/interface | Coarse WIT-facing responsibility | Native owner retained | Decisions required before versioning |
| --- | --- | --- | --- | --- |
| Compute graph execution and tensor resources | Existing `magnetar:compute/run`; possible tensor, graph, and execution boundaries are conceptual, not proposed interface names | Validated fixed-width descriptors, opaque tensor/graph/operation resources, coarse submit/await/cancel | Allocation, storage, kernels, device contexts, synchronization | Graph representation; resource sharing; numerical semantics; evolve `run` as one coarse contract or register separate Capability IDs; interface-to-Capability mapping; package/version migration |
| Model loading | `magnetar:model/load` | Load from a host-authorized artifact resource; return opaque model plus metadata, fingerprint, and abilities | Artifact cache, format detection, weights, mmap, placement, loader implementation | Artifact identity and permissions; ability metadata; affinity and unload lifecycle |
| Tokenization | `magnetar:tokenization/tokenizer` | Open a fingerprinted tokenizer; encode/decode; incremental decoder; special-token metadata | Native tokenizer implementation and cache, when used | Normalization/offset semantics; limits; incremental decode; model fingerprint compatibility |
| Prompt formatting | `magnetar:prompt/chat-template` | Render structured messages/tools/reasoning policy with a fingerprinted template | Template storage and sandbox policy | Message/tool schema; escaping; determinism; template trust and limits |
| Causal generation | `magnetar:generation/causal-lm` | Start/cancel a session and pull ordered token/text/usage/completion/error events | Forward loop, KV cache, batching, sampling, loaded model/device | Sampling semantics; stop conditions; event schema; backpressure; deterministic replay |
| Text completion facade | `magnetar:text/completion` | Text request/result and ability-specific streaming session | Imported tokenizer/generation implementations | Whether this is only a Component export; usage/finish schema; prompt treatment |
| Chat/conversation | `magnetar:chat/conversation` | Structured messages, tools, history policy, streamed assistant response | Imported prompt/tokenization/generation implementations | Component-backed Capability resolution; history ownership; tool and reasoning schemas |
| Speech recognition | `magnetar:speech-recognition/recognize` | Media input resource/chunks, language options, partial/final transcripts | Decode/resample, model execution, native media buffers | Media resource contract; timestamps/confidence; language negotiation; partial-result guarantees |
| Speech synthesis | `magnetar:speech-synthesis/synthesize` | Text/voice request, feature negotiation, audio metadata/chunk session | Model execution, codecs, native audio buffers | Voice identity/cloning policy; reference media authorization; format and chunk guarantees |
| OCR | `magnetar:ocr/recognize` | Image resource, text, optional regions/confidence, optional result stream | Image decode/preprocess and model execution | Coordinate system; region optionality; page/batch semantics; feature negotiation |
| Vision-language | `magnetar:vision-language/generate` | Image resources plus structured messages, generation policy, response events | Image encoder, model/KV state, Compute | Stable input parts; multi-image limits; placeholder source gaps; reuse of generation events |

The current runtime derives a `CapabilityId` from each imported WIT interface.
Consequently, an import such as `magnetar:compute/tensor` would request that
Capability ID and would not resolve to the existing `magnetar:compute/run`
registration. A follow-up must therefore either evolve `run` as one coarse
interface or deliberately register and version separate Capability IDs; this
change does not choose names that the current resolver would treat as real.

Candidate-to-source evidence index:

- Compute graph execution: Candle
  [`Tensor`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/tensor.rs),
  [`Op`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/op.rs), and
  [`BackendStorage`](https://github.com/huggingface/candle/blob/2a13b0f3ff62f7e67013597f2996f764c5735e21/candle-core/src/backend.rs).
- Model loading: Crane
  [`LlmClient::new`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/llm/client.rs) and
  [`model_factory::create_backend`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-serve/src/engine/model_factory.rs).
- Tokenization and prompt formatting: Crane
  [`AutoTokenizer`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-core/src/autotokenizer.rs) and
  [`ChatTemplateProcessor`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-serve/src/chat_template.rs).
- Causal generation: Crane
  [`ModelForCausalLM`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-core/src/generation/based.rs),
  [`ModelBackend`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-serve/src/engine/backend.rs), and
  [`Sequence`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-serve/src/engine/sequence.rs).
- Text completion and chat: Crane
  [`LlmClient`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/llm/client.rs) and
  [`ChatClient`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/chat/client.rs).
- Speech recognition and synthesis: Crane [`Asr`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/audio/asr.rs)
  and [`Tts`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/audio/tts.rs).
- OCR and provisional vision-language: Crane
  [`OcrClient`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane/src/vision/ocr.rs) and server
  [`VlmRequest`](https://github.com/lucasjinreal/Crane/blob/a47b11ce9d36f269d3c100e1f84716b3dbf23777/crane-serve/src/handlers/vlm.rs).

A possible shared `magnetar:media` package may define host-managed image/audio
resources and descriptors, but it is not yet a Capability candidate. Sharing a
resource type across packages couples their versioning and must be designed
before ASR, TTS, OCR, or vision-language contracts are finalized.

`GenerationConfig` is a request record, and streaming is a session/event
pattern. Neither is a standalone Capability. Similarly, individual tensor
operation categories are candidate coverage within Compute, not automatically
independently resolved Provider contracts.

### Native responsibility map

| Native surface | Magnetar owner | Allowed portable projection |
| --- | --- | --- |
| Hardware discovery, physical identity, backend handles | Provider discovers; existing `Device` records identity/metadata | Stable device/capability metadata where host policy permits; never a vendor handle |
| Queue/stream/context, seed state, synchronization | Provider plus scheduler | Await/cancel on a coarse operation or session, not device-wide synchronization |
| Allocation, pools, raw buffers, layout aliases, host staging | Provider plus future memory planner | Opaque tensor/media resources and explicit upload/download/copy operations |
| Kernel dispatch, backend storage, custom Rust ops | Provider behind Compute | Semantic graph/batch operations with structured errors |
| Autograd/training graph | Out of scope for the inference runtime | None in this taxonomy |
| Artifact fetching/cache, paths, credentials, mmap, weight formats | Host artifact service and model Provider | Authorized content-addressed artifact resource plus digest/metadata |
| Loaded weights, forward step, KV cache, cache swap, batching, sampling hot loop | Model Provider plus scheduler | Opaque model/generation-session resources and semantic request/events |
| Tokenizer implementation state and incremental decoder buffer | Tokenization implementation | Opaque tokenizer/decoder resources plus token/text values |
| Image/audio decode, preprocess/resample, codecs, large buffers | Provider or host media service | Host-managed media resources, descriptors, and bounded chunks |
| Rust callbacks, trait objects, channels, locks, `Arc`, generic associated types | Native adapters only | Pull-based resource methods, records, variants, lists, and structured errors |

This map prevents a native implementation detail from being advertised as a
Capability merely to make it discoverable. Provider metadata advertises the
WIT-backed semantic contract; its internal implementation remains private.

### Coarse Component boundary rules

Future contract changes should apply these rules consistently:

1. Use opaque host resources for tensors, models, generation state, images,
   audio, and incremental decoders. Never pass Rust objects or GPU pointers.
2. Make upload, download, copy, materialization, and cross-device movement
   explicit. Do not hide CPU staging or a synchronization barrier.
3. Submit tensor work as a validated graph/batch or another coarse unit, not a
   WIT call for each eager arithmetic primitive.
4. Represent long-running work as a session/operation resource with bounded
   `next`-style pulls, cancellation, ordered events, and a terminal state. Batch
   small token/audio deltas to avoid a Canonical ABI crossing per element.
5. Keep artifact references content-addressed and host-authorized. Portable
   Components never receive ambient filesystem paths, cache directories, or
   service credentials.
6. Use fixed-width values, explicit units, limits, feature negotiation, and
   stable error variants. Backend error strings remain diagnostics, not the
   contract.
7. Attach Provider, Device, artifact, tokenizer, and template affinity to
   opaque resources so dependent calls cannot accidentally resolve an
   incompatible implementation.
8. Let application Components orchestrate lower-level Capabilities; they do not
   select a concrete Provider or hardware backend.

These are interface shapes, not WIT definitions. Function names, ownership
syntax, async representation, and package versions remain follow-up work.

### Fallback matrix

Fallback is phase-sensitive. The primary classification below describes the
strictest state reached by the candidate; the next columns state when a less
restrictive recovery is valid.

| Candidate | Primary class | Before resource creation or output | After state creation or observable output | Replay and compatibility requirements |
| --- | --- | --- | --- | --- |
| Compute graph execution | Provider-pinned | Transparent Provider selection before tensors/contexts are allocated | Pinned while input/output resources or submitted work belong to a Provider; restartable only from host-replayable immutable inputs | Same graph semantics, dtype/layout/shape support, numerical policy, sufficient memory; explicit copies to the new Provider |
| Model loading | Provider-pinned resource | Transparent retry before a model resource is returned | Loaded model remains pinned; unload and reload to switch | Identical artifact digest/format and compatible device/dtype/memory constraints |
| Tokenization | Transparent for stateless calls; Provider-pinned decoder | Stateless encode/decode may switch when the tokenizer fingerprint is identical | Incremental decoder is pinned, or restartable by replaying its full token history | Exact vocabulary, normalization, special-token, and template-compatible fingerprint |
| Prompt formatting | Transparent | Pure render may switch before returning output when template digest and schema are identical | No live Provider state; retry from the complete structured request | Exact template digest, tool/message schema, reasoning policy, and deterministic rendering |
| Causal generation | Provider-pinned | Provider may be selected/retried before session creation and before any event | KV/RNG/model state pins the session. After a delta, emit `interrupted`; any restart is explicit and may diverge | Replay prompt and accepted prefix, same model/tokenizer/template, sampling policy and seed; sequence numbers prevent duplicate output |
| Text completion | Provider-pinned through generation | Transparent before the first result/event if all imported fingerprints match | Active stream is pinned; non-streaming request is restartable only before a result is returned | Complete text/request plus underlying generation compatibility |
| Chat/conversation | Restartable between turns; Provider-pinned within a turn | The Component may resolve a compatible chain before starting the assistant turn | Active assistant generation is pinned. Conversation state may be replayed for a later explicit retry | Full structured history/tools plus model/tokenizer/template fingerprints; replay policy must avoid duplicated tool effects |
| Speech recognition | Provider-pinned stream | Whole-utterance work is restartable before a transcript is returned | After a partial transcript, switching is not transparent; terminate or explicitly restart from replayable audio | Same media interpretation, language/features, model semantics; audio must remain replayable |
| Speech synthesis | Provider-pinned stream | Restartable before the first audio chunk | Pinned after emission; voice, codec, format, or waveform continuity cannot change mid-stream | Same voice/model/codec and generation policy; authorized reference audio remains available |
| OCR | Restartable request; Provider-pinned stream | Retry before returning a document when image input is replayable | Streaming output pins the operation; a backend without region support is not a compatible fallback | Same OCR feature set, coordinate system, language/model semantics, and replayable image |
| Vision-language | Provider-pinned | Retry before session/output when all image/message inputs are replayable | Model/KV state and emitted text pin execution; explicit restart only | Same model/tokenizer/template, image preprocessing, sampling policy, seed, and replayable media |

Provider availability is not execution failover. The current runtime resolves a
deterministically ordered list of compatible Providers, but
`Runtime::resolve_component_import` does not execute a call, monitor health, or
migrate state. It also has no resource-affinity chain across Capability
dependencies. Scheduler work must add health/cost policy, affinity, and the
phase rules above before claiming automatic runtime fallback for these
stateful candidates.
