# Define Post-Baseline Model Format Roadmap

## Why

The first Magnetar baseline intentionally uses fixture model artifacts.

That keeps the first executable path small, deterministic, local, CPU-only, and
conformance-friendly.

After the baseline, Magnetar needs to ingest real-world model formats.

Those formats may include:

- safetensors
- sharded safetensors
- Hugging Face-style config.json
- tokenizer.json
- tokenizer_config.json
- generation_config.json
- chat templates
- SentencePiece
- GGUF
- LoRA adapters
- quantized artifacts
- model index files
- license/provenance metadata

This must be done without breaking the existing architecture.

Model format support SHALL be an artifact ingestion, validation, and
normalization concern.

It SHALL NOT turn model formats into Providers, Runtime capabilities with
ambient filesystem access, or architecture-specific shortcuts.

## What Changes

This change defines the post-baseline roadmap for model format support.

It introduces roadmap phases for:

- normalized Model Artifact manifest
- safetensors weight support
- sharded weight support
- Hugging Face-style config support
- tokenizer.json support
- tokenizer_config support
- generation_config support
- chat template support
- SentencePiece support
- GGUF support
- adapter artifact formats
- quantized model artifact metadata
- license/provenance metadata
- format conformance fixtures

The exact implementation order may vary, but each format SHALL normalize into
the existing Model Artifact, Tokenizer, Adapter, Tensor, and Model Loading
contracts.

## Model Format Roadmap Principle

Model format support SHALL normalize external files into internal Magnetar
artifact contracts.

The canonical flow is:

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

External file formats SHALL NOT bypass Model Artifact validation.

## Format Support Is Not Provider Support

Adding a model format SHALL NOT add a Provider.

Invalid:

```text
GGUFProvider
SafetensorsProvider
QwenSafetensorsProvider
```

Correct:

```text
GGUF / safetensors parser
    -> normalized Model Artifact
    -> Model Component
    -> Execution Graph
    -> Kernel Registry
    -> Provider
```

## Format Support Is Not Model Component Support

A format parser SHALL not own model architecture behavior.

Architecture behavior remains owned by Model Component.

A parser may extract metadata needed by Model Component validation, but it SHALL
not produce Provider-specific execution behavior.

## Phase 1: Normalized Manifest

The first post-baseline format phase SHOULD define a normalized Magnetar Model
Artifact manifest.

The manifest SHOULD support:

- artifact identity
- digest
- architecture family
- model type
- config metadata
- weight files
- tensor inventory
- tokenizer files
- chat template metadata
- generation defaults
- quantization metadata
- adapter metadata
- license metadata
- provenance metadata
- trust metadata
- integrity metadata
- source metadata
- annotations

The normalized manifest SHALL be the canonical input to Model Loading.

## Phase 2: safetensors Support

Magnetar SHOULD support safetensors as a weight format.

Safetensors support SHOULD include:

- metadata parsing
- tensor name inventory
- tensor shape metadata
- tensor dtype metadata
- tensor byte range metadata
- integrity validation
- sharding support placeholder
- memory mapping placeholder
- streaming read placeholder

Safetensors parsing SHALL not expose raw file handles or memory pointers through
public APIs.

## Phase 3: Sharded Weights

Magnetar SHOULD support sharded model weights.

Sharding support SHOULD include:

- index metadata
- shard file list
- tensor-to-shard mapping
- per-shard digest
- total size estimate
- missing shard detection
- duplicate tensor detection
- tensor shape consistency
- loading order policy
- partial loading policy placeholder

Sharded loading SHALL not bypass artifact trust or integrity validation.

## Phase 4: Hugging Face-style Config

Magnetar SHOULD support config metadata commonly found in `config.json`.

Config support SHOULD normalize fields into Model Artifact and Model Component
metadata.

For Qwen-like decoder models, fields MAY include:

- architectures
- model_type
- hidden_size
- num_hidden_layers
- num_attention_heads
- num_key_value_heads
- head_dim
- intermediate_size
- vocab_size
- max_position_embeddings
- rms_norm_eps
- hidden_act
- rope metadata
- tie_word_embeddings
- torch_dtype as source metadata, not authoritative Runtime dtype

Unsupported fields SHALL be preserved as annotations or rejected according to
policy.

## Phase 5: tokenizer.json Support

Magnetar SHOULD support `tokenizer.json` through the Tokenizer Artifact model.

Support SHOULD include:

- tokenizer identity
- vocabulary metadata
- merges/model metadata where available
- normalizer metadata
- pre-tokenizer metadata
- decoder metadata
- added tokens
- special tokens
- encode/decode compatibility
- offset support metadata where available

Tokenizer execution remains owned by Tokenizer Contract.

## Phase 6: tokenizer_config Support

Magnetar SHOULD support tokenizer config metadata.

Support MAY include:

- tokenizer class metadata
- model max length
- padding side
- truncation side
- chat template reference or inline template
- BOS/EOS/PAD token metadata
- added special token metadata
- clean-up tokenization spaces metadata
- source annotations

Tokenizer config SHALL not override Runtime policy silently.

## Phase 7: generation_config Support

Magnetar SHOULD support generation defaults.

Generation config support MAY include:

- max length metadata
- max new tokens metadata
- temperature
- top-k
- top-p
- repetition penalty
- EOS token ID
- BOS token ID
- PAD token ID
- do_sample
- stop strings where present

Generation config values SHALL be defaults, not mandatory Runtime policy.

Runtime Generation API may override them according to policy.

## Phase 8: Chat Template Support

Magnetar SHOULD support chat template metadata through the Prompt Template and
Tokenizer contracts.

Chat template support SHALL include:

- template identity
- source metadata
- compatibility with tokenizer
- compatibility with model family
- variable requirements
- special token interaction
- rendering diagnostics
- raw prompt redaction

Templates SHALL not be fetched from arbitrary filesystem or network during
inference.

## Phase 9: SentencePiece Support

Magnetar MAY support SentencePiece tokenizer artifacts.

Support SHOULD include:

- model identity
- vocabulary size
- special token metadata
- normalization metadata where available
- encode/decode behavior
- compatibility with Tokenizer Contract
- browser support status
- license/provenance metadata

Unsupported SentencePiece features SHALL fail explicitly.

## Phase 10: GGUF Support

Magnetar MAY support GGUF as a post-baseline model artifact format.

GGUF support SHOULD include:

- metadata extraction
- tensor inventory
- tensor shape metadata
- tensor dtype metadata
- quantization metadata
- tokenizer metadata where embedded
- architecture metadata
- alignment and storage metadata
- integrity metadata where available
- memory mapping policy
- conversion into normalized Model Artifact metadata

GGUF support SHALL not create `GGUFProvider`.

GGUF quantized tensors SHALL use Tensor Layout and Quantization metadata.

## Phase 11: Adapter Format Support

Magnetar SHOULD support adapter artifact formats after baseline.

Adapter format support MAY include:

- LoRA safetensors
- adapter_config metadata
- target module metadata
- rank
- alpha
- scaling
- dropout metadata as training/source metadata
- base model compatibility
- tensor inventory
- dtype metadata
- quantization metadata
- license/provenance metadata

Adapter support SHALL normalize into Adapter Artifact contract.

## Phase 12: Quantized Artifact Metadata

Magnetar SHALL handle quantized artifact metadata explicitly.

Quantized metadata SHOULD include:

- quantization method
- bits per value
- group size
- storage dtype
- compute dtype expectation
- scale dtype
- zero point dtype
- packing layout
- tensor-specific quantization metadata
- dequantization requirements
- Provider/Kernel compatibility requirements

No hidden quantization or dequantization SHALL occur.

## Source And Distribution Boundary

Model format support SHALL not imply arbitrary download behavior.

External sources SHALL use the existing Component/Artifact distribution and
source validation contracts.

The Runtime SHALL not perform arbitrary network downloads during inference.

`magnetar-cli` may later implement user-facing download UX, but Runtime Model
Loading still validates normalized artifacts.

## Local File Boundary

If local file paths are supported, path resolution SHALL occur outside Runtime
or through an explicitly authorized artifact source.

Runtime SHALL not scan arbitrary directories during inference.

A client-provided local artifact source SHALL be normalized and validated before
loading.

## Trust And Integrity

Every supported format SHALL participate in trust and integrity validation.

Validation SHOULD include:

- digest checks
- part/shard checks
- manifest consistency
- metadata consistency
- tensor inventory consistency
- tokenizer compatibility
- license/provenance policy
- signature status where available
- revocation status where available

A parser SHALL not mark an artifact trusted by format alone.

## Format Normalization

Format parsers SHALL normalize external metadata into stable Magnetar metadata.

Normalized metadata SHOULD preserve source annotations without making them
authoritative unless validated.

Example:

```text
torch_dtype
```

may be preserved as source metadata but SHALL not silently force Runtime compute
dtype.

## Format Conformance

Each supported format SHALL have conformance fixtures.

Fixtures SHOULD include:

- valid minimal artifact
- missing required metadata
- invalid tensor shape
- invalid dtype
- invalid shard index
- missing shard
- duplicate tensor
- tokenizer mismatch
- unsupported quantization
- malformed file metadata
- untrusted artifact
- redaction checks

## Error Model

Model format roadmap errors SHALL be structured.

Error categories SHOULD include:

- model-format-unsupported
- model-format-invalid
- model-format-parser-failed
- model-manifest-invalid
- model-manifest-missing
- model-config-invalid
- safetensors-invalid
- safetensors-tensor-missing
- safetensors-dtype-unsupported
- shard-index-invalid
- shard-missing
- shard-digest-mismatch
- tokenizer-json-invalid
- tokenizer-config-invalid
- generation-config-invalid
- chat-template-invalid
- sentencepiece-unsupported
- gguf-invalid
- gguf-quantization-unsupported
- adapter-format-invalid
- quantization-metadata-invalid
- model-format-trust-denied
- model-format-integrity-failed
- model-format-local-file-denied
- model-format-network-denied
- internal-model-format-error

## Observability

Runtime SHOULD emit observations for:

- model format detected
- manifest normalized
- manifest validation failed
- config parsed
- config validation failed
- tensor inventory parsed
- tensor inventory mismatch
- tokenizer metadata parsed
- tokenizer compatibility failed
- generation config parsed
- chat template parsed
- safetensors parsed
- shard index parsed
- shard missing
- GGUF metadata parsed
- quantization metadata parsed
- adapter metadata parsed
- integrity validation failed
- trust validation failed

Observability SHALL not expose raw model weights, raw tokenizer data, raw prompts,
raw file contents, secrets, filesystem authority, raw tensor values, memory
pointers, Provider handles, Device handles, or Kernel handles by default.

## Non-Goals

This change does not:

- implement model downloads
- define model hub UX
- define CLI model pull behavior
- implement all model formats immediately
- guarantee large production model execution
- define Provider behavior
- define Qwen architecture semantics
- define training/fine-tuning
- expose raw file handles
- expose raw mmap pointers
- bypass Model Artifact validation
- bypass Model Loading
- bypass trust and integrity checks

## Impact

Magnetar gains a safe roadmap for real-world model format support.

The path becomes:

```text
real model files
    |
    v
format parser / normalizer
    |
    v
normalized Model Artifact
    |
    v
Model Loading
    |
    v
Model Component
    |
    v
Runtime inference
```

without breaking Runtime boundaries.