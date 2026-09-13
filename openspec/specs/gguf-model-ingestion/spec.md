# gguf-model-ingestion Specification

## Purpose
TBD - created by archiving change wire-gguf-into-model-loading. Update Purpose after archive.
## Requirements
### Requirement: GGUF Ingestion Is Scoped To A Real Model Component's Architecture

The GGUF ingestor SHALL accept only GGUF files declaring `general.architecture` values a real Magnetar Model Component exists for (`qwen2`), and SHALL reject any other declared architecture with a structured error naming it, rather than guessing at an unsupported graph shape.

#### Scenario: Supported architecture ingests

- **WHEN** a GGUF file declares `general.architecture: "qwen2"`
- **THEN** ingestion proceeds to normalize architecture configuration and tensors

#### Scenario: Unsupported architecture is rejected

- **WHEN** a GGUF file declares an architecture other than `qwen2`
- **THEN** ingestion returns a structured error naming the declared architecture, and no tensor materialization is attempted

---

### Requirement: GGUF Tensor Shape And Layout Match The Runtime's Existing Convention

The GGUF ingestor SHALL normalize each tensor's shape and, where applicable, byte layout to exactly the same convention the existing Hugging Face/Safetensors ingestion path produces for the equivalent tensor, so the Runtime and Model Component require no format-specific handling.

#### Scenario: A projection weight's logical shape matches the Hugging Face convention

- **WHEN** a GGUF file's 2D projection tensor (`q`/`k`/`v`/output attention projections, MLP gate/up/down projections, or the output/`lm_head` projection) is normalized
- **THEN** its resulting shape and byte layout are `[in_features, out_features]`, identical in convention to what Hugging Face/Safetensors ingestion of the equivalent checkpoint would produce

#### Scenario: The embedding table is never transposed

- **WHEN** a GGUF file's token embedding tensor is normalized
- **THEN** its resulting shape is `[vocab_size, hidden_size]`, matching Hugging Face/Safetensors ingestion's identical treatment of the same tensor as a lookup table, not a projection

#### Scenario: Identical real weight values ingested through either format produce identical generation

- **WHEN** the same real (non-trivial) weight values are ingested once through GGUF ingestion and once through Hugging Face/Safetensors ingestion, and generation is run against identical input token ids on both
- **THEN** the generated token ids are identical

---

### Requirement: A Tied-Embedding GGUF Checkpoint's Output Projection Is Usable Either Way

Whether a GGUF file declares an explicit output/`lm_head` projection tensor or omits it for a tied-embedding checkpoint, the GGUF ingestor SHALL produce a usable `lm_head` tensor entry: the declared one if present, or one derived from the token embedding tensor if absent.

#### Scenario: An explicit output projection is used directly

- **WHEN** a GGUF file declares its own output projection tensor
- **THEN** ingestion uses that tensor's own real bytes for `lm_head`, without deriving a synthetic one

#### Scenario: A missing output projection is derived from the token embedding

- **WHEN** a GGUF file declares no output projection tensor
- **THEN** ingestion derives a `lm_head` tensor from the token embedding tensor's own real bytes, transposed to the same logical shape a declared projection would have

---

### Requirement: GGUF Ingestion Builds A Real Tokenizer From The File's Own Embedded Vocabulary

The GGUF ingestor's tokenizer construction SHALL build a real tokenizer directly from the GGUF file's own embedded vocabulary and merge rules, without requiring a separate tokenizer file, and SHALL satisfy the same special-token uniqueness constraint every tokenizer in the Runtime satisfies regardless of source format.

#### Scenario: A real tokenizer builds from embedded vocabulary alone

- **WHEN** a GGUF file declares a byte-level BPE tokenizer model with embedded token and merge arrays
- **THEN** a real tokenizer builds from those arrays alone, without reading any file other than the GGUF file itself

#### Scenario: A shared beginning-of-sequence and end-of-sequence id does not fail construction

- **WHEN** a GGUF file declares the same token id for both its beginning-of-sequence and end-of-sequence special tokens
- **THEN** tokenizer construction succeeds, registering that id under one role rather than failing on a special-token conflict

### Requirement: GGUF Ingestion Dequantizes Supported Block Formats

The GGUF ingestor SHALL dequantize a tensor whose declared storage uses a supported GGUF block quantization format (`Q8_0`, `Q4_K`, or `Q5_K`) to `F32` before that tensor's shape/layout normalization (including any projection-weight transpose) runs, and SHALL report that tensor's dequantized `F32` storage dtype, byte size, and absence of quantization metadata in the returned Model Artifact -- never the original quantized values. Any GGUF `ggml_type` this ingestor's underlying parser does not itself recognize continues to fail at the parser layer, before this ingestor ever sees such a tensor.

#### Scenario: A supported quantized tensor dequantizes before materialization

- **WHEN** a GGUF file declares a tensor with a supported quantized `ggml_type` (`Q4_K`, `Q5_K`, or `Q8_0`)
- **THEN** the returned tensor's storage dtype is `F32`, its byte size matches its dequantized element count, it carries no quantization metadata, and reading its payload returns the real dequantized values

#### Scenario: A quantized projection weight transposes correctly

- **WHEN** a supported quantized tensor is also a 2D projection weight subject to this ingestor's existing projection-weight transpose
- **THEN** dequantization completes before the transpose runs, and the transposed result matches what transposing the equivalent already-`F32` values would produce

#### Scenario: Unquantized tensors of a mixed file still ingest

- **WHEN** a GGUF file's tensors are entirely unquantized (`F32`/`F16`/`BF16`)
- **THEN** ingestion proceeds normally, unaffected by dequantization support existing for other tensors

