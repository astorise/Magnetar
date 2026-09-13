## REMOVED Requirements

### Requirement: GGUF Ingestion Rejects Quantized Tensors Structurally

**Reason**: Superseded by real dequantization support for the three GGUF block formats `formats/gguf` already recognizes (`Q8_0`/`Q4_K`/`Q5_K`) -- rejecting them unconditionally was a deliberate, explicitly-temporary scope decision (`wire-gguf-into-model-loading`'s own design.md non-goal), not a permanent architectural boundary.

**Migration**: A caller relying on the old rejection behavior for one of these three formats now receives a successfully-ingested, dequantized-to-`F32` tensor instead of an error -- see "GGUF Ingestion Dequantizes Supported Block Formats" below. Any GGUF `ggml_type` this ingestor's underlying parser does not itself recognize is unaffected: it continues to fail at the parser layer, before this ingestor ever sees such a tensor.

## ADDED Requirements

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
