## Decisions

### The layer layout drives validation

A layer's expected tensors and shapes come from its `LayerSpec` when the configuration has a layout:
- **Projections:** a layer with `kv_source_layer` needs no K or V projection, and a layer with `value_from_key` no V projection.
- **Widths:** `q_proj`, `o_proj`, the biases and the MLP take the layer's head width, KV head count and feed-forward width.

A layer that computes its own K/V still needs both projections, so a missing one is still refused.

### The artifact's inventory completes the weight shapes

The shapes derived from the configuration stay as they are. Each tensor of the artifact that they do not name is added with the shape its inventory declares, so the graph-builder offers exactly what weight materialization binds.

A per-family list of optional tensors was not used, for two reasons:
- such a list would put a family's names into the Runtime core;
- it would need updating for every new family.

When the configuration and the inventory both name a tensor, the derived shape is kept. Inventory validation has checked the two against each other.

The configuration alone still serves the paths that have no artifact (fixtures, and segment graphs built from a configuration).

### `moe.*` names for routed experts

The router and stacked experts are `layers.N.moe.router_proj` and `layers.N.moe.experts_{gate,up,down}_proj`, as both loaders, the weight shapes and inventory validation already name them. A Component requesting other names would need a translation table in the Runtime. A per-expert scale follows the same convention: `layers.N.moe.per_expert_scale`.

## Evidence

- **Unit tests:**
  - a layered inventory without the projections its layers do not have passes;
  - a missing projection that a layer does need is refused;
  - a layer sized with the architecture-wide width is refused;
  - the artifact's extra tensors reach the weight shapes.
- **Logits:** the Component's comparison with `transformers` was run against this change, without the Component's local patch. Setup:
  - the Component's `tools/qualification` at `2e0ab1e`;
  - tiny random Gemma 4 bundles;
  - the Reference CPU Provider, f32;
  - 12 greedy tokens.

  The tokens are identical, and the maximum logit differences are 3.5e-5 (dense) and 5.5e-5 (shared KV). Without this change, the same run stops at `qwen tensor inventory missing: layers.2.self_attn.v_proj`.

  Not run: real checkpoints, other Providers, and the per-layer-input and routed-expert variants, which the loaders still refuse.
