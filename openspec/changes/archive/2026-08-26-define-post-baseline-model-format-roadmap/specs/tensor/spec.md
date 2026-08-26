## ADDED Requirements

### Requirement: Model Formats Produce Tensor Metadata

Model format parsers SHALL produce Tensor Descriptor-compatible metadata for
weights and adapter tensors.

#### Scenario: safetensors tensor

Given safetensors tensor has shape and dtype

When normalized

Then Tensor Descriptor metadata can be created without exposing raw storage.

---

### Requirement: Quantized Formats Use Tensor Layout Metadata

Quantized model formats SHALL represent packing and quantization through Tensor
Layout and quantization metadata.

#### Scenario: GGUF quantized tensor

Given GGUF tensor uses quantized packed layout

When normalized

Then Tensor metadata declares packed quantized layout without hidden
dequantization.