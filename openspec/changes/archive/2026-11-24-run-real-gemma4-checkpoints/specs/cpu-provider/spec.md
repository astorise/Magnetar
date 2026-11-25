## ADDED Requirements

### Requirement: CPU Providers Run The Layered And Latent-Attention Forms

The Reference CPU Provider and the optimized CPU Provider of `providers/cpu` SHALL implement and advertise the optional operator forms the Runtime's own Reference CPU implements: the embedding `scale`; per-head and weightless `rmsnorm`; partial and YaRN `rope`; the attention `scale` and `value_head_dimension`; `concat` by heads; `moe-experts` with its activation and per-expert scale; `row-scale`; the `sigmoid`, `softcap` and `scale` activations; and a broadcasting `mul`. The optimized CPU Provider's own attention and embedding Kernels SHALL honour the forms they read, or leave the invocation to the Reference CPU Kernel.

#### Scenario: Tiny checkpoints against transformers
- **WHEN** the tiny Gemma 4 and DeepSeek-V2 checkpoints run on either CPU Provider, the optimized one with f32 activations
- **THEN** every step's top-8 logits match the `transformers` references within 1e-3, with identical tokens
