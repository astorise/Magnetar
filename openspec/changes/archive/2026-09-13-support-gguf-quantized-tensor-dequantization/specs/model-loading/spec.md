## MODIFIED Requirements

### Requirement: Quantization Handling

Loading SHALL explicitly handle quantized artifacts. For a quantization format Loading knows how to dequantize, Loading SHALL convert quantized storage bytes to `F32` compute content explicitly during weight materialization -- exactly as it already does for `F16`/`BF16` storage -- rather than requiring a Provider to consume quantized bytes directly or leaving the format unhandled.

#### Scenario: Unsupported quantization

Given a Model Artifact uses unsupported quantization format

When loading is requested

Then Runtime rejects loading with quantization-unsupported.

#### Scenario: GGUF K-quant/block format dequantizes to F32 at load time

Given a Model Artifact declares a tensor using a supported GGUF block quantization format (`Q8_0`, `Q4_K`, or `Q5_K`)

When weight materialization runs

Then the tensor's quantized storage bytes are converted to `F32` content explicitly, matching the declared tensor shape, and every downstream consumer of that weight receives only `F32` content

#### Scenario: A declared content digest is checked against the original quantized bytes

Given a supported GGUF block-quantized tensor declares a content digest

When weight materialization runs

Then the digest is verified against the original quantized storage bytes before conversion, not against the converted `F32` representation
