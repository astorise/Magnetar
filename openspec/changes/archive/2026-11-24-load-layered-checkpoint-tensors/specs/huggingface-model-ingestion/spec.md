## ADDED Requirements

### Requirement: Hugging Face Ingestion Derives A Layered Decoder's Packed Tensors

Hugging Face ingestion SHALL derive the tensors a layered decoder's Model Component requests from its packed tensors:
- **Per-layer input tables:** each layer's slice of `embed_tokens_per_layer` (columns) and `per_layer_model_projection` (rows).
- **Routed experts:** the stacked gate and up projections from `experts.gate_up_proj`, and the stacked down projection from `experts.down_proj`.
- **Router:** `router.proj` with `router.scale` and `hidden^-0.5` folded in, as `f32`, and `router.per_expert_scale`.

It SHALL refuse packed tensors that disagree with `config.json`, and SHALL fold no factor into any other weight.

#### Scenario: Per-layer embedding slice
- **WHEN** a bundle packs 2 layers of 2 per-layer input values into `embed_tokens_per_layer`
- **THEN** `layers.1.per_layer_embedding` holds columns 2 and 3 of every row

#### Scenario: Experts that disagree with the configuration
- **WHEN** `experts.gate_up_proj` holds a different expert count from `config.json`'s
- **THEN** ingestion fails with malformed metadata
