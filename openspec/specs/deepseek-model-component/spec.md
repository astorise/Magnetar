# deepseek-model-component Specification

## Purpose
This specification defines the DeepSeek Model Component: how it is shipped under a pinned digest and how its generations are qualified against transformers, on Hugging Face bundles, deepseek2 GGUF files and CUDA.
## Requirements
### Requirement: The DeepSeek Model Component Is Shipped Under A Pinned Digest

Magnetar SHALL ship the DeepSeek Model Component built from the pinned `components/deepseek` commit as `deepseek-real.component.wasm`, with a manifest that declares its digest, its `model-component-graph` `1.7.0` imports (`graph-builder` and `model-config` only), the architecture families `deepseek`, `deepseek_v2`, `deepseek_v3` and `deepseek2`, and the artifact formats `huggingface` and `gguf`.

#### Scenario: A DeepSeek-V2 bundle selects the Component
- **WHEN** a `deepseek_v2` Hugging Face bundle is ingested and loaded with the real Components registered
- **THEN** the Component whose manifest declares `deepseek_v2` is the DeepSeek Component, and it loads under its digest

### Requirement: The DeepSeek Model Component Matches transformers On The Reference CPU

The DeepSeek Model Component run by the Reference CPU Provider SHALL generate the same greedy tokens as `transformers`' `DeepseekV2ForCausalLM` in float32 on the checked-in tiny DeepSeek-V2 bundles, with every step's top-8 logits within 1e-3 of the reference.

#### Scenario: Latent attention with YaRN
- **WHEN** a bundle with latent attention, YaRN, a dense first layer and routed and shared experts generates 12 tokens
- **THEN** the tokens are the reference's and the logits are within 1e-3

#### Scenario: A low-rank query with scaled rotations and routing
- **WHEN** a bundle with a low-rank query, YaRN scaling the cosines and sines, and a routed scaling factor of 2.5 generates 12 tokens
- **THEN** the tokens are the reference's and the logits are within 1e-3

### Requirement: The DeepSeek Model Component Matches transformers On deepseek2 GGUF Files

The DeepSeek Model Component SHALL generate, from `deepseek2` GGUF files ingested by the GGUF loader, the same greedy tokens as `transformers`' `DeepseekV2ForCausalLM` computes on the files' dequantized weights, with every step's top-8 logits within 1e-3, whether the latent attention's up-projection is one `attn_kv_b` matrix or split into `attn_k_b` and `attn_v_b`.

#### Scenario: One key-value up-projection
- **WHEN** a `deepseek2` file holding `attn_kv_b`, with the head widths in `key_length`/`value_length` and no YaRN betas, generates 12 tokens on the Reference CPU
- **THEN** the tokens are the reference's and the logits are within 1e-3

#### Scenario: Split key and value up-projections
- **WHEN** a `deepseek2` file holding a quantized `attn_k_b` (each head's key rows transposed) and `attn_v_b`, with `head_count_kv` 1 and the head widths in `key_length_mla`/`value_length_mla`, generates 12 tokens on the Reference CPU
- **THEN** the tokens are the reference's and the logits are within 1e-3

#### Scenario: On CUDA
- **WHEN** the tiny bundles and `deepseek2` files generate on CUDA with f32 activations
- **THEN** the tokens are the references' and the logits are within 1e-3

### Requirement: DeepSeek-V2-Lite Generates On CUDA

Magnetar SHALL load the published DeepSeek-V2-Lite-Chat `Q8_0` GGUF file, pinned by its revision and checked against its declared digest, across two CUDA Devices with the DeepSeek Model Component, and SHALL continue "The capital of France is" greedily with "Paris".

#### Scenario: The capital of France
- **WHEN** the file is loaded with `MAGNETAR_CUDA_DEVICES` listing two GPUs and generates greedily from "The capital of France is"
- **THEN** the generated text starts with "Paris"
