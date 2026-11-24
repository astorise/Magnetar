## Why

The Model Component of issue #187 ran a layered checkpoint through the production path and compared its logits with `transformers`. The bundle ingested, but loading failed in three places that still assume a uniform decoder:

1. Tensor inventory validation requires `self_attn.k_proj` and `self_attn.v_proj` in every layer. A layer that reuses an earlier layer's K/V has neither, and a layer deriving V from K has no V projection.
2. Tensor shape validation sizes every layer with the architecture-wide head width and feed-forward width, so a full layer with wider heads is refused.
3. The weight shapes offered to the graph-builder are derived from the architecture configuration only. The per-head norms, the extra norms and the layer scale are absent, although weight materialization binds them from the artifact.

## What changes

- Inventory and shape validation follow each layer's entry in the layer layout, when the configuration declares one.
- A loaded Model Artifact's graph session is also offered every tensor of the artifact that the configuration does not derive, with the shape its inventory declares. The production loading paths use it.

The routed experts keep their `moe.*` logical names, which both loaders, the weight shapes and inventory validation already use.

## Impact

- `magnetar-runtime`: `validate_first_native_tensor_inventory`, `first_native_expected_tensor_shape`, `weight_shapes_for_artifact`, and `build_first_native_graphs_from_named_component_for_artifact`, which the production loading and execution paths call.
- Uniform configurations are unaffected: without a layout, validation and the derived weight shapes are unchanged.
