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

Magnetar SHOULD support safetensors metadata and tensor inventory parsing, and parsed tensors SHALL normalize into Model Artifact tensor metadata.

#### Scenario: Tensor listed

Given safetensors file contains tensor metadata

When parser runs

Then tensor name, shape, dtype, and storage metadata are normalized.

---

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

Magnetar MAY support GGUF normalization into Model Artifact metadata, and GGUF support SHALL NOT create `GGUFProvider`.

#### Scenario: GGUF parsed

Given GGUF artifact is parsed

When normalization completes

Then tensor, architecture, tokenizer, and quantization metadata are available
without creating GGUFProvider.

---

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

