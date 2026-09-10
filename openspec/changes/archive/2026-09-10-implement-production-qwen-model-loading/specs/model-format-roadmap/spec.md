## ADDED Requirements

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
