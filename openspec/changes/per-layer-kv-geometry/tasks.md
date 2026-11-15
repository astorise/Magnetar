## 1. Configuration

- [x] 1.1 `FirstNativeModelConfig`: `layer_layout`, `final_logit_softcapping`, `per_layer_input_size`, carried through both conversions; `kv_row_widths()`.

## 2. Engine

- [x] 2.1 `attention` dispatch: the node's geometry, `window_size` and `scale`; `scale` gated by `attention-scale`, advertised by the Reference CPU.
- [x] 2.2 Pooled store bytes and released stores per producing layer.
- [x] 2.3 `KvCacheLayoutMetadata::layer_row_widths` in the KV cache estimate.
- [x] 2.4 The weight shapes a Component's graph session gets follow each layer's geometry, without the K/V projections a layer lacks.

## 3. Tests

- [x] 3.1 A layered graph (sliding window and GQA, full attention with `scale`, shared KV) against an independent host computation, prefill and decode, history concatenated and in place.
- [x] 3.2 `kv_row_widths`, the per-layer estimate, per-layer weight shapes, and the feature check.

## 4. Validation

- [x] 4.1 `cargo fmt`, `clippy -D warnings`, and tests for `magnetar-core`, `magnetar-model`, `magnetar-engine` and `magnetar-runtime`.
