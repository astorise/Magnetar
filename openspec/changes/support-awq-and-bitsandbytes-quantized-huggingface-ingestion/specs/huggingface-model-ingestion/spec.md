## ADDED Requirements

### Requirement: Hugging Face Ingestion Dequantizes Supported AWQ Projections

The Hugging Face ingestor SHALL dequantize a 4-bit AWQ (GEMM-kernel) quantized projection (a raw tensor triple sharing one module prefix: `<prefix>.qweight`, `<prefix>.qzeros`, `<prefix>.scales`, with no `<prefix>.g_idx` sibling) to `F32`, at its already Runtime-oriented `[in_features, out_features]` logical shape, before that projection's shape/layout normalization (including this ingestor's own `nn.Linear`-storage-convention transpose) runs, and SHALL report that projection's dequantized `F32` storage dtype, byte size, and absence of quantization metadata in the returned Model Artifact -- never the original packed values. A bit width other than 4 SHALL be rejected with a structured error rather than silently mis-decoded.

#### Scenario: An AWQ-quantized projection dequantizes before materialization

- **WHEN** a Hugging Face bundle declares a projection as an AWQ-quantized `.qweight`/`.qzeros`/`.scales` triple with no `.g_idx` sibling
- **THEN** the returned tensor's storage dtype is `F32`, its shape is `[in_features, out_features]`, it carries no quantization metadata, and reading its payload returns the real dequantized values

#### Scenario: An AWQ-dequantized projection is not transposed a second time

- **WHEN** an AWQ-quantized projection is also a name this ingestor's existing `nn.Linear`-storage-convention transpose would otherwise match
- **THEN** dequantization's own already-correct `[in_features, out_features]` orientation is preserved, not swapped again

#### Scenario: A GPTQ-quantized bundle is not misinterpreted as AWQ

- **GIVEN** a `.qweight` tensor with a `.g_idx` sibling (GPTQ's own disambiguating signal)
- **WHEN** the bundle is ingested
- **THEN** it is dequantized as GPTQ, not AWQ, regardless of the two schemes' otherwise-identical `.qweight`/`.qzeros`/`.scales` naming

---

### Requirement: Hugging Face Ingestion Dequantizes Supported BitsAndBytes Projections

The Hugging Face ingestor SHALL dequantize a 4-bit BitsAndBytes (NF4/FP4) quantized projection -- identified by a `<prefix>.weight` tensor with a `<prefix>.weight.absmax` sibling, and a real `<prefix>.weight.quant_state.bitsandbytes__nf4` JSON blob declaring the projection's true logical shape and block size(s) -- to `F32`, at its real `[out_features, in_features]` logical shape (`nn.Linear`'s own storage convention), before that projection's shape/layout normalization runs, and SHALL report that projection's dequantized `F32` storage dtype, byte size, and absence of quantization metadata in the returned Model Artifact. A codebook other than 16 values (i.e. not 4-bit) SHALL be rejected with a structured error rather than silently mis-decoded. Both single-level and double-quantized (`absmax` itself quantized) encodings SHALL be supported.

#### Scenario: A BitsAndBytes-quantized projection dequantizes before materialization

- **WHEN** a Hugging Face bundle declares a projection as a BitsAndBytes-quantized `.weight` tensor with an `.absmax` sibling and a real `quant_state` JSON blob
- **THEN** the returned tensor's storage dtype is `F32`, its shape matches the real logical shape the `quant_state` blob declares, it carries no quantization metadata, and reading its payload returns the real dequantized values

#### Scenario: A BitsAndBytes-dequantized projection still transposes normally

- **WHEN** a BitsAndBytes-quantized projection is also a name this ingestor's existing `nn.Linear`-storage-convention transpose matches
- **THEN** it is transposed the same way a plain, unquantized weight of the same name would be -- unlike a GPTQ- or AWQ-dequantized projection, which is excluded from that transpose

#### Scenario: A bundle with no quantized tensors of any supported scheme ingests unaffected

- **WHEN** a Hugging Face bundle's tensors are entirely unquantized
- **THEN** ingestion proceeds normally, unaffected by GPTQ/AWQ/BitsAndBytes dequantization support existing for other bundles
