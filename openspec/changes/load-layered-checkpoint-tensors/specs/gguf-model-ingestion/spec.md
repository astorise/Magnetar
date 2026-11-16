## ADDED Requirements

### Requirement: GGUF Ingestion Derives A Layered File's Packed Tensors

GGUF ingestion SHALL derive the tensors a `gemma4` file's Model Component requests from its packed tensors:
- **Per-layer input tables:** each layer's slice of `per_layer_token_embd` (columns) and `per_layer_model_proj` (rows).
- **Routed experts:** the stacked gate and up projections from `ffn_gate_up_exps`.
- **Router:** `ffn_gate_inp` with `ffn_gate_inp.scale` and `hidden^-0.5` folded in, as `f32`, and `ffn_down_exps.scale` as the per-expert scale.

A block-quantized table SHALL be cut on block boundaries and copied without re-encoding, and a cut inside a block SHALL be refused.

#### Scenario: Block-quantized per-layer embedding
- **WHEN** `per_layer_token_embd` is `Q8_0` with one 32-element block per layer per row
- **THEN** `layers.1.per_layer_embedding` is each row's second block, byte for byte
