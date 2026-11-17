## ADDED Requirements

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
