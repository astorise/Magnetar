## ADDED Requirements

### Requirement: Production Qwen Profile Uses A Real Tokenizer Artifact Implementation

The production Qwen loading profile SHALL provide a real tokenizer implementation behind Tokenizer Contract capable of consuming normalized `tokenizer.json` plus relevant tokenizer configuration/special-token metadata.

#### Scenario: Prompt encoded for loaded model
- **WHEN** a production Qwen Model Instance has a compatible loaded tokenizer
- **AND** a caller submits text for generation
- **THEN** Tokenizer Contract encodes that text using the real tokenizer artifact rather than a deterministic fixture tokenizer.

#### Scenario: Tokenizer vocabulary mismatch
- **WHEN** the tokenizer vocabulary metadata is incompatible with the loaded model configuration
- **THEN** loading/session creation fails with a structured tokenizer compatibility error before generation.

---

### Requirement: Production Tokenizer Supports Decode And Streaming Decode

The production tokenizer implementation SHALL support normal decode and incremental/streaming decode according to Tokenizer Contract.

#### Scenario: Generated tokens streamed
- **WHEN** Generation produces token IDs incrementally
- **THEN** the tokenizer can emit valid text chunks while preserving pending partial byte/tokenization state according to the contract.

---

### Requirement: Generation Config Is Default Metadata

Values normalized from `generation_config.json` SHALL be defaults and SHALL NOT override an explicit caller generation request or Runtime policy silently.

#### Scenario: Caller overrides temperature
- **WHEN** the artifact default temperature differs from an explicit accepted request temperature
- **THEN** the explicit request value is used.

---

### Requirement: Chat Template Is Artifact-Bound

A production chat template used for Qwen inference SHALL come from Runtime-authorized artifact/config data and SHALL NOT trigger arbitrary filesystem or network access during inference.

#### Scenario: Template path points outside artifact
- **WHEN** source metadata attempts to reference an unauthorized external template path
- **THEN** normalization/loading rejects it or treats it as non-authoritative metadata rather than fetching it during inference.
