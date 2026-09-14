## ADDED Requirements

### Requirement: Hugging Face Ingestion Dequantizes Supported GPTQ Projections

The Hugging Face ingestor SHALL dequantize a 4-bit GPTQ-quantized projection (a raw tensor quadruple sharing one module prefix: `<prefix>.qweight`, `<prefix>.qzeros`, `<prefix>.scales`, `<prefix>.g_idx`) to `F32`, at its already Runtime-oriented `[in_features, out_features]` logical shape, before that projection's shape/layout normalization (including this ingestor's own `nn.Linear`-storage-convention transpose) runs, and SHALL report that projection's dequantized `F32` storage dtype, byte size, and absence of quantization metadata in the returned Model Artifact -- never the original packed values. A bit width other than 4 SHALL be rejected with a structured error rather than silently mis-decoded.

#### Scenario: A GPTQ-quantized projection dequantizes before materialization

- **WHEN** a Hugging Face bundle declares a projection as a GPTQ-quantized tensor quadruple
- **THEN** the returned tensor's storage dtype is `F32`, its shape is `[in_features, out_features]`, it carries no quantization metadata, and reading its payload returns the real dequantized values

#### Scenario: A GPTQ-dequantized projection is not transposed a second time

- **WHEN** a GPTQ-quantized projection is also a name this ingestor's existing `nn.Linear`-storage-convention transpose would otherwise match
- **THEN** dequantization's own already-correct `[in_features, out_features]` orientation is preserved, not swapped again

#### Scenario: An unsupported GPTQ bit width fails closed

- **WHEN** a GPTQ-quantized projection's declared packing implies a bit width other than 4
- **THEN** ingestion fails with a structured error naming the unsupported bit width, rather than producing silently incorrect dequantized values

#### Scenario: A bundle with no GPTQ tensors ingests unaffected

- **WHEN** a Hugging Face bundle's tensors are entirely unquantized
- **THEN** ingestion proceeds normally, unaffected by GPTQ dequantization support existing for other bundles
