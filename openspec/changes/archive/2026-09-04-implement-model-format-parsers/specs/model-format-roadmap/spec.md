## MODIFIED Requirements

### Requirement: Safetensors Support
Magnetar SHALL support real Safetensors parsing (header, tensor inventory, byte ranges) with overflow and panic safety on untrusted input, and parsed tensors SHALL normalize into Model Artifact tensor metadata. This requirement's normative behavior is defined in full by the `safetensors-format` capability; this entry records that Safetensors support is implemented, not aspirational roadmap language.

#### Scenario: Tensor listed
- **WHEN** a Safetensors file contains tensor metadata
- **THEN** tensor name, shape, dtype, and storage metadata are normalized into `ModelTensorMetadata`.

#### Scenario: Malformed header rejected
- **WHEN** a Safetensors file has a truncated or invalid JSON header
- **THEN** the parser returns a structured error rather than panicking.

### Requirement: GGUF Support
Magnetar SHALL support real GGUF parsing (chunked metadata, tensor info) with overflow and panic safety on untrusted input, for the dtype/quantization subset `ModelDType`/`ModelQuantizationFormat` model (`Q4K`, `Q5K`, `Q8`, unquantized float/integer types); tensors using other quantization types are explicitly rejected rather than silently mis-mapped. GGUF support SHALL NOT create `GGUFProvider`. This requirement's normative behavior is defined in full by the `gguf-format` capability; this entry records that GGUF support is implemented for its supported subset, not aspirational roadmap language, and that full quantization-type coverage remains real follow-up work.

#### Scenario: GGUF parsed
- **WHEN** a GGUF artifact using a supported dtype/quantization is parsed
- **THEN** tensor and quantization metadata are available without creating `GGUFProvider`.

#### Scenario: Unsupported quantization rejected
- **WHEN** a GGUF artifact declares a tensor using a quantization type outside the supported subset
- **THEN** the parser returns a structured `gguf-quantization-unsupported` error rather than approximating it.
