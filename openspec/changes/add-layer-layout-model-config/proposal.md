## Why

Issue #187 asks the Runtime to run models whose decoder layers differ from one another: sliding-window or full attention, a per-layer head width and KV head count, a RoPE base per attention kind, layers that reuse an earlier layer's K/V, and per-layer feed-forward widths. Some of these models also cap their logits and add per-layer input embeddings.

`model-config@1.5.0` describes one uniform layer only, so a Model Component cannot learn any of this. It has to come from the Runtime, which reads the checkpoint, and not from the Component, which must never recompute a per-layer rule.

The Model Component for this family is already written against `1.6.0`. Its WIT matches the host's `1.5.0` WIT except for the additions below.

## What changes

- **WIT `magnetar:model-component-graph@1.6.0`:** these additions go in `interface model-config`. Nothing existing changes.
  - The `layer-spec` record holds:
    - `sliding-window`;
    - `head-dimension`;
    - `num-key-value-heads`;
    - `rope-theta`;
    - `partial-rotary-factor`;
    - `value-from-key`;
    - `kv-source-layer`;
    - `intermediate-size`.
  - `layer-layout: func() -> option<list<layer-spec>>`.
  - The `architecture-options` record holds `final-logit-softcapping` and `per-layer-input-size`.
  - `options: func() -> architecture-options`.
- **`ModelArchitectureConfig`:**
  - It gains `layer_layout: Option<Vec<LayerSpec>>`, `final_logit_softcapping` and `per_layer_input_size`. All three are `None` for every checkpoint loaded today.
  - `validate()` refuses an inconsistent layout or option.
- **`ModelConfigCapability`:** it answers `layer-layout` and `options` with the bound values.
- **Version registration:** the generic Component registry also provides `1.6.0`. It keeps `1.2.0` to `1.5.0`, so Components compiled against them still link.
- **Loaders:** the GGUF and Hugging Face loaders leave the new fields at their defaults.

Not in this change:
- deriving a layout from a checkpoint (the loader mappings);
- per-layer KV geometry and shared KV in the first-native engine;
- dispatching the layered forms;
- the Provider Kernels.

Each is a later step on issue #187.

## Impact

- **`magnetar-model`:** the `LayerSpec` type, the new config fields and their validation.
- **`magnetar-engine`:** the `layer-layout` and `options` operations, and a WAT probe test that crosses the canonical ABI.
- **`magnetar-runtime`:** the WIT file and the `1.6.0` registration.
- **`loaders/gguf`, `loaders/huggingface`:** `..ModelArchitectureConfig::default()` for the new fields.

Existing Components and checkpoints behave exactly as before.
