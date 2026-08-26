# Post-Baseline Model Format Roadmap

The first Magnetar baseline intentionally uses fixture model artifacts:
small, deterministic, local, CPU-only, and conformance-friendly. This roadmap
defines how Magnetar ingests real-world model formats -- safetensors,
sharded safetensors, Hugging Face-style `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `generation_config.json`, chat templates,
SentencePiece, GGUF, LoRA adapters, and quantized artifacts -- without
turning any of them into a Provider, a Runtime capability with ambient
filesystem access, or an architecture-specific shortcut.

This document, and the `magnetar-runtime::model_format_roadmap` module it
describes, do **not** implement byte-level safetensors/GGUF/SentencePiece
parsers, model downloads, model hub UX, or CLI pull behavior. They define the
roadmap **contract** -- phases, normalization targets, structured errors,
observability categories, and conformance checks -- that any future format
ingestion work must satisfy.

## Model Format Roadmap Principle

Model format support is an artifact ingestion, validation, and normalization
concern. The canonical flow is:

```text
external model files
    |
    v
format parser / manifest builder
    |
    v
normalized Model Artifact
    |
    v
Model Loading
    |
    v
Model Component compatibility
    |
    v
Tensor Resource / Memory Manager
```

External file formats never bypass Model Artifact validation.

## Format Support Is Not Provider Support

Adding a model format never adds a Provider (`GGUFProvider`,
`SafetensorsProvider`, `QwenSafetensorsProvider`-shaped names are all
invalid). `reject_model_format_provider_name` implements this as an
executable, regression-proof check -- mirroring
[`reject_model_family_provider_name`](provider-roadmap.md) but keyed on
format fragments (`gguf`, `safetensors`, `sentencepiece`, `tokenizerjson`,
...) instead of model-family names. Hardware/optimized Provider names such as
`ReferenceCpuProvider` or `CudaProvider` are unaffected.

`reject_format_execution_graph` implements the companion rule: a format
parser may only ever produce normalized metadata, never an authoritative
execution graph -- that stays owned by Model Component.

## Roadmap Phases

`ModelFormatRoadmapPhase` enumerates the twelve post-baseline phases from the
proposal, in `SHOULD`-order:

1. normalized Model Artifact manifest
2. safetensors weight support
3. sharded weight support
4. Hugging Face-style config support
5. tokenizer.json support
6. tokenizer_config support
7. generation_config support
8. chat template support
9. SentencePiece support
10. GGUF support
11. adapter format support
12. quantized artifact metadata

Every phase's `normalizes_into_existing_contract()` is `true`: no phase
introduces a parallel artifact type. This roadmap reuses the existing Model
Artifact, Tokenizer, and Adapter contracts throughout.

## Normalized Manifest

Rather than introducing a parallel manifest type, [`NormalizedManifestCoverage`]
reports which roadmap-named fields an already-validated `ModelManifest`
actually carries (identity, digest, architecture, weight files, tensor
inventory, tokenizer, chat template, generation defaults, quantization,
license, provenance, source) -- proving the *existing* `ModelManifest`, which
is already the canonical input to Model Loading, is the roadmap's normalized
manifest.

## safetensors

`SafetensorsManifest` carries a tensor inventory (`SafetensorsTensorEntry`:
name, shape, dtype, byte range) and free-form header metadata. `validate()`
rejects empty/duplicate names, degenerate shapes, and zero-length tensors.
`into_tensor_metadata()` normalizes into `ModelTensorMetadata`, the existing
tensor inventory contract. Neither type has a field through which a raw file
handle or memory pointer could be represented, so "Safetensors parsing SHALL
not expose raw file handles or memory pointers" holds structurally.
`MemoryMappingPolicy::validate()` additionally rejects any policy that claims
to expose a raw pointer.

## Sharded Weights

`ShardIndex` reuses the existing `ModelShard`/`ModelShardId` contract instead
of a parallel sharding type. `detect_missing_shards` catches a tensor
referencing an absent shard; `detect_duplicate_tensor_names` catches the same
tensor name appearing under more than one shard;
`validate_shard_tensor_shape_consistency` catches the same tensor name
declaring different shapes; `validate_shard_loading_order` requires strictly
increasing shard order. None of these bypass artifact trust or integrity
validation -- they run as part of normalization, before Model Loading.

## Hugging Face-style Config

`HfConfigMetadata` carries the Qwen-like decoder fields the proposal names
(`hidden_size`, `num_attention_heads`, `num_key_value_heads`, `rope`,
`tie_word_embeddings`, ...) plus `torch_dtype` and free-form `annotations`.
`normalize_architecture()` converts into the existing `ModelArchitecture`
contract. `torch_dtype_does_not_force_compute_dtype` makes "torch_dtype is
preserved as source metadata but SHALL not silently force Runtime compute
dtype" checkable: the function accepts a `torch_dtype` and structurally
ignores it, always returning the caller's `requested_compute_dtype`.

## tokenizer.json

`TokenizerJsonMetadata` carries vocabulary size, added/special tokens,
normalizer/pre-tokenizer/decoder metadata, and offset support.
`normalize_tokenizer_json` converts it into `TokenizerMetadata`, the existing
Tokenizer Artifact contract the Tokenizer Contract already consumes --
tokenizer execution itself remains owned by that contract, not by this
module.

## tokenizer_config

`TokenizerConfigMetadata` carries tokenizer class, model max length, padding
and truncation side, chat template reference, BOS/EOS/PAD token metadata,
added special tokens, and clean-up-tokenization-spaces. Because "tokenizer
config SHALL not override Runtime policy silently",
`reject_silent_tokenizer_config_override` requires an explicit
`runtime_policy_validated` attestation before a parsed value can become
effective policy -- otherwise it is source annotation only.

## generation_config

`GenerationConfigMetadata` carries max length/new tokens, temperature,
top-k/top-p, repetition penalty, EOS/BOS/PAD token IDs, `do_sample`, and stop
strings. `as_defaults()` converts into the existing `ModelGenerationDefaults`
contract. `apply_generation_override` makes "generation_config values SHALL
be defaults, not mandatory Runtime policy" checkable: an explicit request
value always wins over the parsed default.

## Chat Template Support

`ChatTemplateSourceKind` has exactly three variants -- embedded in manifest,
authorized local artifact, client-provided inline -- and deliberately no
`Http`/`RemoteUrl`/`ArbitraryFilesystem` variant, so "Templates SHALL not be
fetched from arbitrary filesystem or network during inference" holds because
no such source can even be constructed. `validate_chat_template` requires
tokenizer and model-family compatibility and every declared required
variable to be present at render time. `redact_chat_template_diagnostic`
reuses the existing `redact_backend_diagnostic` path for raw prompt
redaction.

## SentencePiece Support

`SentencePieceMetadata` carries model identity, vocabulary size, special
tokens, normalization, browser support status, and license/provenance.
`reject_unsupported_sentencepiece_feature` makes "unsupported SentencePiece
features SHALL fail explicitly" checkable against a declared
`supported_features` set.

## GGUF Support

`GgufMetadata` carries architecture, alignment, a tensor inventory
(`GgufTensorEntry`, each optionally carrying `ModelQuantization`), embedded
tokenizer metadata, and free-form key-values. `into_tensor_metadata()`
normalizes into the same `ModelTensorMetadata` contract safetensors
normalizes into -- "GGUF quantized tensors SHALL use Tensor Layout and
Quantization metadata", not a GGUF-specific tensor type. "GGUF support SHALL
not create `GGUFProvider`" is enforced by
`reject_model_format_provider_name`, which explicitly covers this name.

## Adapter Format Support

`LoraAdapterFormatMetadata` carries target modules, rank, alpha, scaling,
dropout (preserved as training/source metadata only), base model
compatibility, tensor inventory, dtype, quantization, and license/provenance.
`normalize_lora_adapter` converts it into the existing `AdapterArtifact`
contract, always setting `trust: AdapterTrustStatus::Unknown` -- "parsing an
adapter format SHALL not activate the adapter" and format alone never grants
trust either; both require a separate, explicit policy decision downstream.

## Quantized Artifact Metadata

`ModelFormatQuantizationDeclaration` composes the existing
`ModelQuantization` weight-storage contract with optional
`KernelQuantizationMetadata` Provider/Kernel compatibility metadata (defined
in `kernel.rs`, shared with the [provider roadmap](provider-roadmap.md)).
`validate_model_format_quantization` requires a declared scale dtype and, when
kernel compatibility is declared, delegates to the existing
`provider_roadmap::validate_quantization_declaration` and
`reject_hidden_dequantization` so quantization metadata and dequantization
behavior are validated by one shared rule set -- "no hidden quantization or
dequantization SHALL occur" holds for model-format-declared quantization
exactly as it does for Kernel-declared quantization.

## Source, Local File, Trust, and Integrity Boundaries

`reject_arbitrary_model_download` documents that `ModelArtifactSource` is a
closed, already-authorized enum (local path, local cache, client-provided,
registry, Hugging Face, OCI, Tachyon) -- there is no variant through which an
arbitrary download could be represented. `reject_raw_network_model_reference`
additionally denies a raw `http(s)://`/`ftp://` string used as a model
reference. `validate_local_file_boundary` denies a `LocalPath` source unless
it is explicitly attested as authorized -- Runtime never scans arbitrary
local directories. `model_format_grants_no_trust` always evaluates trust
through the existing `ModelTrustStore`, never inferring it from which parser
produced the manifest: "a model format SHALL not be trusted merely because it
is recognized."

## Format Conformance

`ModelFormatConformanceFixtureKind` names the twelve fixture categories from
the proposal (valid minimal artifact, missing required metadata, invalid
tensor shape, invalid dtype, invalid shard index, missing shard, duplicate
tensor, tokenizer mismatch, unsupported quantization, malformed file
metadata, untrusted artifact, redaction check).

## Error Model

`ModelFormatRoadmapError` covers every error category from the proposal: the
generic `model-format-unsupported`/`-invalid`/`-parser-failed` categories,
`model-manifest-invalid`/`-missing`, `model-config-invalid`, the three
`safetensors-*` categories, the three `shard-*` categories,
`tokenizer-json-invalid`, `tokenizer-config-invalid`,
`generation-config-invalid`, `chat-template-invalid`,
`sentencepiece-unsupported`, `gguf-invalid`,
`gguf-quantization-unsupported`, `adapter-format-invalid`,
`quantization-metadata-invalid`, `model-format-trust-denied`,
`model-format-integrity-failed`, `model-format-local-file-denied`,
`model-format-network-denied`, and `internal-model-format-error`.

## Observability

`ModelFormatRoadmapObservationKind` covers all nineteen categories from the
proposal. `ModelFormatRoadmapObservation` carries only an observation kind,
an optional artifact identity string, and a `redacted_metadata` string map
whose values are always passed through `redact_backend_diagnostic` before
being stored -- there is no field through which a raw model weight, raw
tokenizer data, raw prompt, raw file content, secret, filesystem path, raw
tensor value, memory pointer, or Provider/Device/Kernel handle could reach
the observation.

## Conformance Report

`run_model_format_roadmap_conformance` produces a
`ModelFormatRoadmapConformanceReport` (mirroring
`ProviderRoadmapConformanceReport`) asserting: format-shaped Provider names
are rejected while hardware/optimized names are allowed; format parsers
cannot supply execution graphs; missing, duplicate, and inconsistent shard
tensors are detected; `torch_dtype` never forces Runtime compute dtype;
`generation_config` defaults are always overridable by an explicit request;
chat template rendering requires compatibility and every required variable;
unsupported SentencePiece features fail explicitly; hidden dequantization is
rejected; unauthorized local file access and raw network model references
are both denied; and an unrecognized digest is never trusted merely because
a parser accepted the file.

## Local Commands

Run the model format roadmap tests:

```powershell
cargo test -p magnetar-runtime model_format_roadmap -- --nocapture
```

Run the full Runtime suite:

```powershell
cargo test --workspace --all-targets
```

Validate the OpenSpec change:

```powershell
openspec validate define-post-baseline-model-format-roadmap --strict
```

## Compatibility Versioning

The current roadmap contract version is `0.1.0`, exposed as
`MODEL_FORMAT_ROADMAP_VERSION`. Passing this contract's conformance checks
does not imply any real-world format parser has been implemented -- it only
confirms the roadmap's structural guarantees (no format Providers, no
parser-owned execution graphs, explicit shard/quantization/trust
declarations, source-metadata-is-not-policy, deny-by-default local/network
access) hold in this Runtime revision.
