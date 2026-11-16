## Why

A layer that reuses an earlier layer's K/V (`kv_source_layer`, issue #187) reads that layer's KV history at every step. The multi-Device layer placement could put a segment boundary between the two layers. The multi-GPU CUDA Provider would then copy the source's whole KV history to the reader's GPU each time the history grew, at every decode step.

## What changes

`LayerPlacement::plan` keeps every layer that reuses an earlier layer's K/V in that source layer's segment:
- `LayerWeightBytes::joined` marks the layers no segment may start at, and `keep_kv_sources_together` sets it from `(layer, source)` pairs.
- A segment reaching its balanced target at such a layer grows to the next allowed start. A segment reaching its Device's budget ends at the last allowed start.
- The production placement takes the pairs from the artifact's layer layout.

## Impact

- `magnetar-provider-api` (`layer_placement`) and `magnetar-runtime` (production layer placement).
- Layers without a layout, or without K/V reuse, are placed as before.
