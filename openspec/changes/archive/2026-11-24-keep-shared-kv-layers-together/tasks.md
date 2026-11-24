## 1. Placement

- [x] 1.1 `LayerWeightBytes::joined` and `keep_kv_sources_together`.
- [x] 1.2 `LayerPlacement::plan`: no segment starts at a joined layer; grow at the target, shrink at the budget.
- [x] 1.3 Production placement takes the K/V sources from the layer layout (`layer_placement_bytes`).

## 2. Tests

- [x] 2.1 Balanced, budget-bound and non-fitting placements with joined layers.
- [x] 2.2 The production placement marks the layers from a layout.
