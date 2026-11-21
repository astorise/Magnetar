## ADDED Requirements

### Requirement: The Gemma 4 Model Component Is Shipped Under A Pinned Digest

Magnetar SHALL ship the Gemma 4 Model Component built from the pinned `components/gemma` commit as `gemma-real.component.wasm`, with a manifest that declares its digest, its `model-component-graph` `1.8.0` imports (`graph-builder` and `model-config` only), the architecture families `gemma4`, `gemma4_text` and `gemma4_unified`, and the artifact formats `huggingface` and `gguf`. The Component SHALL refuse an unknown or unsupported checkpoint activation.

#### Scenario: A Gemma 4 checkpoint selects the Component
- **WHEN** a Gemma 4 Hugging Face bundle or `gemma4` GGUF file is ingested and loaded with the real Components registered
- **THEN** the Component whose manifest declares the checkpoint's family is the Gemma Component, and it loads under its digest

#### Scenario: The graphs it builds pass the Runtime's validation
- **WHEN** the artifact runs in the Wasmtime engine with the real `graph-builder` and `model-config` Capabilities and the published 31B, 26B-A4B or E2B layout
- **THEN** it builds a prefill graph the graph-builder accepts, and a segment that splits a shared-KV layer from its producer is refused

### Requirement: The Gemma 4 Model Component Matches transformers On The Reference CPU

The Gemma 4 Model Component run by the Reference CPU Provider SHALL generate the same greedy tokens as `transformers`' `Gemma4ForCausalLM` in float32 on the checked-in tiny checkpoints, with every step's top-8 logits within 1e-3 of the reference.

#### Scenario: Hugging Face bundles
- **WHEN** the dense bundle and the `gemma4_unified` bundle generate 12 tokens from the reference prompt
- **THEN** the tokens are the reference's and the logits are within 1e-3

#### Scenario: Quantized GGUF files
- **WHEN** the `Q4_0`/`Q8_0` dense, shared-KV, per-layer-input and MoE GGUF files generate 12 tokens
- **THEN** the tokens and logits match the reference computed on their dequantized weights
