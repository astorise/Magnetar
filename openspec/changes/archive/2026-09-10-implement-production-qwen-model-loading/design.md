## Context

The current architecture already contains most of the hard execution-side invariants needed for production loading:

- Model Artifact identity/manifest/tensor/shard/dtype/quantization types exist in `magnetar-runtime`.
- Model Loading validates trust, compatibility, residency, allocation, lifecycle, cleanup, and Model Instance readiness.
- Weight materialization is transactional and provider-backed.
- First-native execution is generic over Runtime `Provider` / `Device` contracts and has real CUDA E2E evidence.
- `formats/safetensors` and `formats/gguf` are real external parsers.
- The strict Qwen graph source is a real external WebAssembly Component.

What is missing is the production composition layer. A real Safetensors parser cannot by itself create an authoritative `ModelManifest`: a weight file does not carry the full model identity, source authorization, config, tokenizer, generation defaults, or trust decision. Conversely, the Core must not solve this by importing Hugging Face/Safetensors libraries because `project-architecture` requires concrete Model Artifact format implementations to stay external.

The implementation therefore needs two boundaries:

1. a generic Core-owned ingestion/payload contract;
2. an external Hugging Face/Qwen ingestor implementing it.

The Runtime remains the only authority that can turn normalized artifact data into loaded inference state.

## Goals / Non-Goals

**Goals:**

- Load a real authorized Hugging Face-style Qwen bundle from local/client-provided source into a ready `ModelInstance`.
- Parse and normalize `config.json`, tokenizer/config/generation/chat-template metadata, single-file or sharded Safetensors.
- Support F16/BF16 storage with explicit conversion to a supported compute dtype.
- Materialize weights incrementally/transactionally instead of requiring a whole-model host tensor map.
- Make Qwen graph production depend on Runtime-authorized normalized model configuration rather than fixture constants.
- Execute the loaded Model Instance through Reference CPU and CUDA without a `qwen-test` privilege.
- Preserve trust, source authorization, Memory Manager, Provider/Device, Component authority, rollback, and readiness invariants.

**Non-Goals:**

- Remote hub/OCI/cache implementation.
- Native half-precision CUDA compute.
- Full GGUF/quantized execution.
- Other architecture families.
- Format-specific types inside Core public loading state.
- Components parsing raw model files.

## Decisions

### Decision 1 — Core defines `ModelArtifactIngestor`-shaped contracts; concrete ingestors stay external

The exact Rust names are implementation details, but Core SHALL expose a generic interface whose semantic output is:

```text
Normalized ModelManifest
+ normalized optional tokenizer/config/generation/chat metadata
+ authorized bounded access to artifact/tensor payload bytes
```

A concrete implementation SHALL NOT be imported by `magnetar-runtime`.

Recommended first module:

```text
loaders/huggingface
repository: astorise/Magnetar-loader-HuggingFace
```

It may depend on public `magnetar-runtime` contracts and `Magnetar-format-safetensors`. The reverse dependency is forbidden.

An embedder registers/composes an implementation; Runtime still performs trust, memory, component, residency, materialization, and readiness decisions.

### Decision 2 — An ingestor normalizes; it never grants trust

Parsing success cannot imply trust.

```text
authorized source
→ ingest/normalize
→ ModelManifest::validate
→ Runtime trust-store evaluation
→ integrity validation
→ component compatibility
→ residency/materialization
```

Digest checks are performed at their authoritative boundaries: artifact/part/shard bytes against declared digests and tensor bytes against tensor digest when declared. A cache/source/format/ingestor hit never skips Runtime trust evaluation.

### Decision 3 — Source authorization precedes filesystem access

A production local bundle is represented by an already-authorized source boundary. The ingestor MAY enumerate files only within that authorized bundle root/handle and SHALL reject path traversal, symlink escape according to chosen policy, and arbitrary sibling-directory scanning.

The Model Component never receives this source capability.

### Decision 4 — Hugging Face bundle normalization produces canonical existing contracts

The first production profile recognizes:

```text
config.json
tokenizer.json
tokenizer_config.json
generation_config.json
model.safetensors
model.safetensors.index.json
model-*-of-*.safetensors
authorized chat-template metadata/file where supported
```

No `HuggingFaceModelInstance`, `SafetensorsModelInstance`, or `QwenSafetensorsProvider` type is introduced. All model execution state still becomes existing `ModelManifest`, `LoadedModelContext`, `ModelInstance`, Tensor Resources and Memory Manager state.

### Decision 5 — `config.json` is normalized before Component graph production

The ingestor parses source-format fields into generic Runtime-authorized model configuration metadata. The Qwen Component consumes the subset it needs through a versioned Capability. It SHALL NOT receive `config.json` bytes/path directly.

Required first-profile values include, where applicable:

```text
model_type
architectures
hidden_size
intermediate_size
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim (or validated derivation)
vocab_size
rms_norm_eps
rope_theta
rope_scaling
tie_word_embeddings
torch_dtype (source annotation)
bos_token_id
eos_token_id
```

The Runtime validates generic structural constraints; the Qwen Component validates Qwen architecture semantics.

### Decision 6 — Extend the Component contract, do not add Runtime-side Qwen graph synthesis

`model-component-graph@1.0.0` cannot make a general Qwen Component configurable because the graph exports receive only sequence/cached-token counts and the current minimal tensor descriptor assumes contiguous Float32.

Introduce a versioned compatible evolution (recommended `magnetar:model-component-graph@1.1.0` if WIT compatibility permits, otherwise `2.0.0`) with Runtime-owned authorized model-config access.

The capability exposes logical configuration values, not raw JSON, arbitrary paths, raw weight bytes, Runtime resource handles, or Provider/Device identity. Strict first-native continues to require Component-produced graphs.

### Decision 7 — Qwen weight names and shapes become configuration-derived

The external Qwen Component SHALL no longer require fixture constants for production graph generation. For `layer in 0..num_hidden_layers`, it derives the required logical weights and output shapes from normalized configuration.

Tied embedding semantics remain logical aliasing/derived load-time state, not file-range aliasing. Any mismatch between declared config and actual tensor inventory fails before ready-state publication.

### Decision 8 — F16/BF16 storage support is required; native half compute is not

The production loader SHALL accept F16/BF16 storage for the first Qwen profile.

If selected Providers only support Float32 compute, loading performs an explicit conversion:

```text
storage F16/BF16 bytes
→ verify source-content digest
→ checked decode/convert
→ F32 staging tensor
→ Provider upload
```

The conversion is part of the Model Residency Plan and observability. It is never inferred silently from `torch_dtype`. Native Provider Float16/BFloat16 kernels are a later optimization.

### Decision 9 — Production materialization is streaming and transactional

The existing whole-map entry point may remain for fixtures/tests, but it SHALL NOT be the only production route.

```text
read tensor payload
→ bounds/dtype/shape/digest validation
→ convert if needed
→ stage Provider resource
→ record pending allocation/binding
→ release host staging
→ next tensor
```

`commit` publishes bindings/readiness only after every required tensor succeeds. On any failure, Provider-side staged tensors, Memory Manager allocations/residency, pending bindings and readiness evidence are all rolled back.

### Decision 10 — Sharded Safetensors index is authoritative mapping metadata, not trust

`model.safetensors.index.json` supplies tensor -> shard mapping and optional total-size metadata. The loader cross-checks it with real Safetensors tensor inventories. Every mapped tensor must be present exactly once, every required shard must exist, and referenced file names must remain within the authorized bundle.

Index presence never grants trust.

### Decision 11 — Tokenizer execution remains behind Tokenizer Contract

The production profile uses a real implementation capable of loading `tokenizer.json` plus relevant `tokenizer_config.json` special-token policy.

At minimum it supports encode, decode, streaming decode, special token handling and model/tokenizer vocabulary compatibility. `generation_config.json` normalizes into defaults that explicit request values can override. A chat template is artifact data, not ambient filesystem/network data at inference time.

### Decision 12 — `ModelInstanceId`, not a magic model name, becomes execution authority

The production caller flow resolves/loads a source to a ready Model Instance, then generation targets that instance.

`qwen-test` remains useful for deterministic conformance and demos but SHALL NOT select a special parser, synthesize a manifest, bypass ingestion, select an architecture config, or unlock an otherwise inaccessible execution path.

### Decision 13 — Tachyon consumes the public production loading surface, not internal primitives

Tachyon SHALL be able to supply an authorized local/client/Tachyon artifact source and receive a load result / Model Instance identity.

Tachyon SHALL NOT need to parse model files, construct Qwen architecture config, map tensor names to shards, convert storage dtypes, allocate/provider-upload model weights, create Runtime Tensor Resource identities, or select Kernels.

If Tachyon needs such logic, the production Magnetar loading boundary is incomplete.

### Decision 14 — Acceptance uses both production-shaped tiny artifacts and a real checkpoint

Per-PR tests need deterministic, small artifacts. They may use a tiny Qwen-compatible architecture, but SHALL use the real production file layout and production parser/orchestrator. They SHALL NOT call fixture-only manifest/inventory constructors.

A separate real-checkpoint smoke test (manual/nightly/hardware-gated) uses a public Qwen-compatible checkpoint pinned by revision/digest and proves CPU execution, CUDA execution on real hardware, no silent hardware skip/fallback, numeric/output compatibility within defined tolerance and no resource leaks.

## Risks / Trade-offs

- **New external module/repository overhead.** Accepted because putting Hugging Face-specific production parsing in Core would violate the existing externalization architecture.
- **Runtime trait/API design can become too I/O-specific.** Keep contracts semantic: authorized source, normalized artifact, bounded payload access. Do not encode `std::fs::File`, mmap, HTTP, or Hugging Face concepts into Core.
- **F32 conversion increases host/VRAM footprint.** Streaming bounds host staging. Native half compute remains a performance follow-up.
- **Tokenizer libraries can be large/native-featureful.** Keep implementation external and gate platform-specific support; Tokenizer Contract remains platform-neutral.
- **WIT contract evolution affects Component compatibility.** Version it explicitly and keep old fixture Component support only where policy permits; production profile requires the configurable version.
- **A tiny production-shaped E2E can still hide scale bugs.** Pair it with at least one real-checkpoint smoke test and large-shard boundary tests.
