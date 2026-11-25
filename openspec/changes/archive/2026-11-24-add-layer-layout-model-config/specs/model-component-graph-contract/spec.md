## ADDED Requirements

### Requirement: Model configuration exposes a per-layer layout

`model-config@1.6.0` SHALL expose `layer-layout`, which returns one `layer-spec` per decoder layer when the active artifact's layers differ, or `none` when `architecture-config` describes every layer. It SHALL also expose `options`, carrying `final-logit-softcapping` and `per-layer-input-size`.

The Runtime SHALL derive these values from the artifact's normalized configuration. It SHALL refuse a layout whose entry count differs from the layer count, whose widths are zero, whose KV heads do not divide the attention heads, or whose `kv-source-layer` is not an earlier layer that computes its own K/V with the same head width and KV head count.

#### Scenario: Uniform checkpoint
- **WHEN** a Component bound to a checkpoint with uniform layers calls `layer-layout` and `options`
- **THEN** it gets `none` and an `architecture-options` record whose fields are both `none`

#### Scenario: Layered checkpoint
- **WHEN** the bound configuration carries a layout and options
- **THEN** `layer-layout` returns one record per layer with the bound values, and `options` returns the bound softcap and per-layer input width

#### Scenario: Inconsistent layout
- **WHEN** a configuration's layout names a later layer as a KV source
- **THEN** validation fails before any Component is invoked

### Requirement: Earlier contract versions keep linking

The Runtime SHALL provide `graph-builder` and `model-config` at `1.6.0` and at every earlier supported version, `1.2.0` to `1.5.0`.

#### Scenario: Component compiled against 1.5.0
- **WHEN** a Component whose manifest declares `1.5.0` is linked after this change
- **THEN** it links and builds graphs as before
