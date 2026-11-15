## Why

Issue #187 targets models whose decoder layers differ:
- sliding or full attention;
- per-layer KV head count and head width;
- layers that attend over an earlier layer's K/V instead of computing their own.

`model-config@1.6.0` now carries that layout (`add-layer-layout-model-config`), but the first-native engine still assumed one KV geometry for the whole model:
- the `attention` dispatch sent the architecture's `head_count`, `kv_head_count` and `head_dimension` whatever the node declared, and dropped its `window_size` and `scale`;
- the KV store pool and the KV cache size estimate counted every layer at the architecture's width;
- the production path converted the configuration through `FirstNativeModelConfig`, which had no layout, so the layout never reached the Component.

## What changes

- **`FirstNativeModelConfig`:**
  - It carries `layer_layout`, `final_logit_softcapping` and `per_layer_input_size`, in both conversions to and from `ModelArchitectureConfig`.
  - `kv_row_widths()` gives each layer's K/V row width, or `None` for a layer that reuses an earlier layer's K/V.
- **`attention` dispatch:**
  - A node's own `head_count`, `kv_head_count`, `head_dimension`, `window_size` and `scale` reach the Kernel. Absent geometry falls back to the architecture's, as before.
  - `scale` is sent only to a Provider whose every active `attention` Kernel advertises the new `attention-scale` feature (`ATTENTION_SCALE_FEATURE`). Otherwise dispatch is refused, so a Kernel never ignores it. The Reference CPU advertises it.
- **KV sizing:**
  - A pooled store set counts the K and V of each producing layer at its own width.
  - `KvCacheLayoutMetadata` gains optional `layer_row_widths`, which `estimated_storage_bytes` uses when set. Released pooled sets skip layers without stores.
- **Weight shapes:** the weight shapes a Component's graph session gets follow each layer's head width, KV heads and feed-forward width. They omit the K/V projections of a layer that reuses another layer's K/V, and the V projection of a layer that derives V from K.
- **Shared KV:**
  - A consumer layer's `attention` node reads the producer's K/V edges.
  - Those edges resolve to the producer's history (concatenated or in place), so a consumer sees the producer's full history.
  - Only producing layers write K/V, so only they hold KV state and stores. This needed no engine change; a test now pins it.

Not in this change:
- deriving a layout from a checkpoint (the loaders);
- the other layered forms' dispatch (`rmsnorm` per head and without weight, partial `rope`, `activation` softcap and scale, scalar `mul`, `gelu` tanh, 6-input `moe-experts`);
- CUDA `scale` support;
- the logits-row gather after a softcap;
- segment planning that keeps a consumer layer in its producer's segment.

## Impact

- **`magnetar-core`:** the `ATTENTION_SCALE_FEATURE` constant.
- **`magnetar-runtime`:**
  - `FirstNativeModelConfig`, `KvCacheLayoutMetadata`, the engine's KV sizing, the `attention` dispatch and the Reference CPU advertisement;
  - tests.

Uniform models behave exactly as before: their nodes either omit the geometry attributes or declare the architecture's values, and their per-layer widths sum to the old estimate.
