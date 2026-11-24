## Why

Layered checkpoints pack some weights in a form the Model Component does not request. The loaders refused these tensors, so neither loader could load Gemma 4 E2B, E4B or 26B-A4B (issue #187):
- **Per-layer input tables:** E2B and E4B pack every layer's per-layer input embedding into one table, and every layer's context projection into another.
- **Routed experts:** 26B-A4B stores them as fused, stacked tensors.
- **Router:** its input scale is a separate tensor.

## What changes

Both loaders now derive the tensors the Component requests, as agreed on #187 (section 3 of the Component's requirements):

| Derived tensor | Hugging Face source | GGUF source |
|---|---|---|
| `layers.N.per_layer_embedding` | columns `[N*P, (N+1)*P)` of `embed_tokens_per_layer` | the same columns of `per_layer_token_embd` |
| `layers.N.per_layer_model_projection` | rows `[N*P, (N+1)*P)` of `per_layer_model_projection` | the same rows of `per_layer_model_proj` |
| `layers.N.moe.experts_gate_proj`, `experts_up_proj` | halves of `experts.gate_up_proj`, stacked | halves of `ffn_gate_up_exps`, stacked |
| `layers.N.moe.experts_down_proj` | `experts.down_proj` | `ffn_down_exps` |
| `layers.N.moe.router_proj` (f32) | `router.proj` × `router.scale` × `hidden^-0.5` | `ffn_gate_inp` × `ffn_gate_inp.scale` × `hidden^-0.5` |
| `layers.N.moe.per_expert_scale` | `router.per_expert_scale` | `ffn_down_exps.scale` |

`P` is the per-layer input width.

No weight is fabricated. The only folded factor is the router's, since a router stays in float.

## Impact

- `loaders/huggingface` and `loaders/gguf`: a new `layered_tensors` module each, partial reads of a stored tensor, and the names of the packed tensors.
- The Runtime is unchanged: it already offers a loaded artifact's tensors to the graph-builder (`validate-layered-artifacts-in-production`).
- Magnetar pins both loaders.
