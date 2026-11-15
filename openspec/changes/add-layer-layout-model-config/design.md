## Context

`model-config` grew additively before:
- `1.2.0` added `attention-bias`;
- `1.4.0` added `rope-scaling`;
- `1.5.0` added `mixture-of-experts`.

Each time, the host kept providing the earlier versions, because a compiled Component's imports are checked against the version its manifest declares. `1.6.0` follows the same pattern.

## Decisions

### The Runtime owns the layout

`layer-layout` returns one `layer-spec` per decoder layer. The Runtime derives every field from the checkpoint; that derivation is the loaders' later step.

The Component reads the layout and never recomputes a per-layer rule, such as "every sixth layer is global". A uniform checkpoint gets `none`. A Component that needs a layout and gets `none` refuses to build a graph.

### Family-neutral fields

`layer-spec` describes what differs between layers, in terms that any architecture with per-layer geometry can use. The WIT and the Rust types never name a model family.

### Validation in `ModelArchitectureConfig::validate`

The Runtime refuses a layout that a Component could not turn into a sound graph:
- the entry count differs from `num_hidden_layers`;
- a head width or feed-forward width is zero;
- the KV heads are zero or do not divide the attention heads;
- the sliding window is zero;
- the RoPE base is non-finite or not positive;
- the rotary fraction is outside `(0, 1]`;
- `kv-source-layer` names a layer that is not earlier, that itself reuses another layer's K/V, or whose head width or KV head count differs.

It also refuses a non-positive softcap and a zero per-layer input width.

### Encoding

`u64` fields are encoded as `S64`, as the earlier `model-config` encoders already do. The Wasmtime value conversion lowers them to the declared `u64`.

A WAT fixture imports `model-config@1.6.0` and reads both functions back through the canonical ABI, with real memory and `realloc`. This shows that the list of records and the options record cross the host-import boundary with the expected values.

### Loader forward compatibility

The loaders, which are separate repositories built inside a Magnetar checkout, fill the new fields with `..ModelArchitectureConfig::default()`. A later field added to the config then does not need another loader change.

## Risks

- **The first-native production path still converts through `FirstNativeModelConfig`, which has no layout.** A layered checkpoint therefore cannot reach a Component through that path yet. The Component then gets `none` and refuses, which is the intended failure until the engine step lands.
