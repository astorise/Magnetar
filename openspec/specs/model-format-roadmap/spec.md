# model-format-roadmap Specification

## Purpose
This specification defines the post-baseline model format roadmap, parser responsibilities, manifest normalization, quantization metadata, and trust boundaries.
## Requirements
### Requirement: Post-Baseline Model Format Roadmap

Magnetar SHALL define a post-baseline roadmap for real-world model format
support.

#### Scenario: Roadmap available

Given fixture baseline is complete

When model format work begins

Then roadmap phases and boundaries are defined.

---

### Requirement: Formats Normalize Into Model Artifact

Model format support SHALL normalize external files into the existing Model
Artifact contract.

#### Scenario: safetensors model

Given safetensors files are parsed

When normalization completes

Then Runtime receives normalized Model Artifact metadata.

---

### Requirement: Format Support Is Not Provider Support

Model formats SHALL not introduce Providers.

#### Scenario: GGUFProvider attempted

Given implementation introduces `GGUFProvider`

When roadmap validation runs

Then validation rejects it.

---

### Requirement: Format Support Is Not Model Component Support

Format parsers SHALL not own model architecture behavior.

#### Scenario: Config parsed

Given parser extracts Qwen-like config fields

When architecture behavior is needed

Then Qwen Model Component validates and interprets architecture semantics.

---

### Requirement: Normalized Manifest

Magnetar SHOULD define a normalized Model Artifact manifest, and Model Loading SHALL treat it as canonical input.

#### Scenario: Manifest created

Given external model files are normalized

When Model Loading runs

Then it consumes normalized manifest metadata.

---

### Requirement: Safetensors Support
Magnetar SHALL support real Safetensors parsing (header, tensor inventory, byte ranges) with overflow and panic safety on untrusted input, and parsed tensors SHALL normalize into Model Artifact tensor metadata. This requirement's normative behavior is defined in full by the `safetensors-format` capability; this entry records that Safetensors support is implemented, not aspirational roadmap language.

#### Scenario: Tensor listed
- **WHEN** a Safetensors file contains tensor metadata
- **THEN** tensor name, shape, dtype, and storage metadata are normalized into `ModelTensorMetadata`.

#### Scenario: Malformed header rejected
- **WHEN** a Safetensors file has a truncated or invalid JSON header
- **THEN** the parser returns a structured error rather than panicking.

### Requirement: Sharded Weight Support

Magnetar SHOULD support sharded weight metadata, and shard validation SHALL detect missing or duplicate shards.

#### Scenario: Missing shard

Given shard index references a missing shard

When validation runs

Then Runtime reports shard-missing.

---

### Requirement: Hugging Face-style Config Support

Magnetar SHOULD normalize common config metadata into Model Artifact and Model Component metadata, and unsupported fields SHALL be preserved as annotations or rejected by policy.

#### Scenario: Qwen config

Given config contains `num_attention_heads`

When normalization runs

Then the value is available for Model Component validation.

---

### Requirement: tokenizer.json Support

Magnetar SHOULD support tokenizer.json metadata, and normalized metadata SHALL conform to the Tokenizer Artifact contract.

#### Scenario: Vocabulary parsed

Given tokenizer.json contains vocabulary metadata

When parser runs

Then Tokenizer Artifact metadata includes vocabulary information.

---

### Requirement: tokenizer_config Support

Magnetar SHOULD support tokenizer_config metadata, and it SHALL NOT silently override Runtime policy.

#### Scenario: Padding side

Given tokenizer_config declares padding side

When normalization runs

Then the value is available as tokenizer metadata or annotation.

---

### Requirement: generation_config Support

Magnetar SHOULD support generation_config, and its values SHALL be treated as defaults, not mandatory Runtime policy.

#### Scenario: Temperature default

Given generation_config declares temperature

When Generation request supplies another temperature

Then Runtime policy determines override behavior.

---

### Requirement: Chat Template Support

Magnetar SHOULD support chat template metadata through authorized prompt and Tokenizer contracts, and templates SHALL NOT be fetched from arbitrary filesystem or network sources.

#### Scenario: Template rendering

Given chat messages are supplied

When Runtime applies template

Then it uses authorized template metadata and not arbitrary filesystem fetch.

---

### Requirement: SentencePiece Support

Magnetar MAY support SentencePiece artifacts through Tokenizer Contract, and unsupported features SHALL fail explicitly.

#### Scenario: Unsupported feature

Given SentencePiece artifact uses unsupported behavior

When parser validates it

Then Runtime returns sentencepiece-unsupported.

---

### Requirement: GGUF Support
Magnetar SHALL support real GGUF parsing (chunked metadata, tensor info) with overflow and panic safety on untrusted input, for the dtype/quantization subset `ModelDType`/`ModelQuantizationFormat` model (`Q4K`, `Q5K`, `Q8`, unquantized float/integer types); tensors using other quantization types are explicitly rejected rather than silently mis-mapped. GGUF support SHALL NOT create `GGUFProvider`. This requirement's normative behavior is defined in full by the `gguf-format` capability; this entry records that GGUF support is implemented for its supported subset, not aspirational roadmap language, and that full quantization-type coverage remains real follow-up work.

#### Scenario: GGUF parsed
- **WHEN** a GGUF artifact using a supported dtype/quantization is parsed
- **THEN** tensor and quantization metadata are available without creating `GGUFProvider`.

#### Scenario: Unsupported quantization rejected
- **WHEN** a GGUF artifact declares a tensor using a quantization type outside the supported subset
- **THEN** the parser returns a structured `gguf-quantization-unsupported` error rather than approximating it.

### Requirement: Adapter Format Support

Magnetar SHOULD support adapter formats, and normalization SHALL produce Adapter Artifact metadata.

#### Scenario: LoRA adapter

Given LoRA safetensors and adapter_config are parsed

When normalization completes

Then Adapter Artifact metadata includes target modules, rank, alpha, and tensor
inventory.

---

### Requirement: Quantized Metadata Explicitness

Quantized artifact metadata SHALL be explicit.

#### Scenario: Quantized tensor

Given tensor uses packed quantized layout

When normalization runs

Then quantization method, group size, dtype, scale, zero-point, and layout
metadata are explicit.

---

### Requirement: Source Boundary

Model format support SHALL not imply arbitrary download behavior.

#### Scenario: Model URL

Given a URL is provided as model reference

When Runtime validates it

Then Runtime uses authorized source contracts or rejects arbitrary network
access.

---

### Requirement: Local File Boundary

Runtime SHALL not scan arbitrary local directories during inference.

#### Scenario: Local path

Given local path is supplied by CLI

When Runtime receives artifact reference

Then it validates explicit authorized artifact source metadata.

---

### Requirement: Trust And Integrity Validation

Every supported format SHALL participate in trust and integrity validation.

#### Scenario: Digest mismatch

Given shard digest does not match manifest

When validation runs

Then Runtime returns model-format-integrity-failed or shard-digest-mismatch.

---

### Requirement: Format Normalization Preserves Source Annotations

Format normalization SHOULD preserve source annotations, and unvalidated annotations SHALL NOT become authoritative Runtime policy.

#### Scenario: torch_dtype

Given config contains `torch_dtype`

When normalized

Then it is preserved as source metadata and does not silently force compute
dtype.

---

### Requirement: Format Conformance

Each supported format SHALL have conformance fixtures.

#### Scenario: Invalid tokenizer

Given tokenizer metadata is incompatible with model metadata

When conformance runs

Then tokenizer mismatch fixture fails as expected.

---

### Requirement: Model Format Error Categories

Model format failures SHALL use structured error categories.

#### Scenario: Invalid safetensors

Given safetensors metadata is malformed

When parser runs

Then Runtime reports safetensors-invalid or model-format-parser-failed.

---

### Requirement: Model Format Observability

Runtime SHOULD emit model format observations, and observations SHALL redact raw weights, file contents, and secrets by default.

#### Scenario: Manifest validation failed

Given manifest validation fails

When observability records it

Then no raw model weights, file contents, secrets, or memory pointers are logged.

### Requirement: Model Formats Integrate With Source Cache

Model format normalization SHALL integrate with source/cache workflow.

#### Scenario: Safetensors from cache

Given safetensors artifact is found in cache

When normalization runs

Then normalized Model Artifact metadata is produced before loading.

---

### Requirement: Format Parser Does Not Own Source Policy

Format parsers SHALL not decide whether a source is allowed or trusted.

#### Scenario: Valid GGUF denied

Given GGUF metadata is parseable

But source policy denies it

When loading runs

Then loading is rejected by policy.

### Requirement: Production Hugging Face Bundle Normalization

Magnetar's first production Qwen loading profile SHALL normalize Hugging Face-style model bundle metadata into existing Model Artifact, Tokenizer, and generation contracts through an external implementation.

At minimum the profile SHALL recognize authorized `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, chat-template metadata where used, and Safetensors weight files/indexes.

#### Scenario: Complete local bundle
- **WHEN** an authorized bundle contains compatible Qwen config, tokenizer metadata, and Safetensors weights
- **THEN** the external ingestor produces canonical normalized metadata consumable by Model Loading
- **AND** no Hugging Face-specific Model Instance or Provider identity is introduced.

#### Scenario: Required file missing
- **WHEN** a required production-profile file is absent
- **THEN** normalization fails with a structured missing/invalid artifact error rather than substituting fixture metadata.

---

### Requirement: Production Hugging Face Config Parsing

The first production Qwen profile SHALL parse real Hugging Face-style `config.json` bytes and normalize architecture fields required for Qwen compatibility and graph production.

Source annotations SHALL NOT silently become Runtime execution policy.

#### Scenario: Valid Qwen config
- **WHEN** `config.json` declares valid Qwen decoder dimensions, attention/KV head counts, RoPE metadata, vocabulary, tied embeddings, and dtype annotations
- **THEN** those values are normalized into Runtime-authorized model configuration metadata.

#### Scenario: Invalid head configuration
- **WHEN** parsed head counts, head dimension, or hidden size are inconsistent
- **THEN** normalization/model compatibility fails with a structured config error before graph execution.

---

### Requirement: Sharded Safetensors Index Parsing

Magnetar's first production Qwen profile SHALL parse Hugging Face-style `model.safetensors.index.json` and validate its tensor-to-shard mapping against the parsed Safetensors shard inventories.

#### Scenario: Index disagrees with shard
- **WHEN** the index maps a tensor to a shard that does not contain that tensor
- **THEN** normalization fails before Model Loading may publish a ready instance.

#### Scenario: Tensor appears in multiple shards
- **WHEN** the normalized shard inventories contain the same logical required tensor more than once
- **THEN** normalization fails with a structured duplicate/inconsistent tensor error.

---

### Requirement: Production Format Normalization Does Not Grant Ambient File Access

A concrete production model ingestor SHALL only read files reachable through an explicitly authorized source/bundle boundary.

#### Scenario: Bundle path escapes root
- **WHEN** an index/config reference attempts to resolve outside the authorized bundle boundary
- **THEN** the ingestor rejects the reference
- **AND** no arbitrary filesystem access is performed.

