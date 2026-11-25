## Context

The engine sizes a KV store from the rows it writes (`output_tensor.rows_cols()`), so a store already takes its layer's width. The assumptions of a single global geometry were in four places:
1. the `attention` dispatch;
2. the pool's byte accounting;
3. the KV cache estimate;
4. the configuration conversion, which dropped the layout.

## Decisions

### The node's attributes are authoritative

The Model Component already writes each layer's geometry on its `attention` node. The dispatch forwards `head_count`, `kv_head_count`, `head_dimension`, `window_size` and `scale` when present, and otherwise keeps the architecture-derived values. The Runtime still derives nothing from a node identifier.

`window_size` needs no gate: the Reference CPU and CUDA attention Kernels both honour it, prefill and decode, with `key_count`.

`scale` is gated like banded RoPE scaling: dispatch is refused unless every active `attention` Kernel of the Provider advertises `attention-scale`. The CUDA attention Kernels do not read `scale` yet. Without the gate they would silently compute with `1/sqrt(head_dimension)`.

### Shared KV needs no new state

A consumer layer has no K/V nodes. Its `attention` node takes the producer's K/V edges as inputs. After the producer's append, those edges are bound to the producer's concatenated history, or to its in-place store with its `key_count`. The consumer therefore reads exactly the producer's history, prefill and decode alike.

KV state, commit and release are keyed by the layers that write K/V, so consumer layers hold none. `kv_row_widths()` returns `None` for them, so neither the pool nor the estimate counts them.

### Per-layer widths in `KvCacheLayoutMetadata`

The metadata stays a single contiguous description, with an optional per-layer width list that replaces `layer_count * head_count * head_dimension` in the estimate. The first-native engine always sets it. For a uniform model it sums to the same value.

## Evidence

`graph_execution::tests::layers_with_their_own_kv_geometry_and_a_shared_kv_layer_prefill_and_decode` builds a three-layer graph on the Reference CPU:
- layer 0: sliding window 2, one KV head of width 4 under two query heads;
- layer 1: full attention with `scale = 0.25` and two KV heads of width 3;
- layer 2: reuses layer 0's K/V.

It checks the following:
- **Prefill logits:** they match an independent host computation within 1e-4.
- **Decode logits:** a decode step over the prefill's history matches the full sequence's last row, with the history concatenated and in place.
- **KV state:** only layers 0 and 1 keep KV state, at widths 4 and 6.
- **Sensitivity:** dropping `window_size` from the dispatch makes it fail (checked by hand during development).
