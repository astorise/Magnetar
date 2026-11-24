## 1. Validation

- [x] 1.1 Inventory validation skips the K/V projections a layer's layout entry says it does not have.
- [x] 1.2 Expected shapes follow each layer's head width, KV head count and feed-forward width.

## 2. Graph-builder weight shapes

- [x] 2.1 `weight_shapes_for_artifact`: the derived shapes, plus the inventory's other tensors.
- [x] 2.2 `build_first_native_graphs_from_named_component_for_artifact`, used by the production loading and execution paths.

## 3. Tests

- [x] 3.1 A layered inventory and its shapes, including the refusals.
- [x] 3.2 The artifact's extra tensors in the weight shapes.
- [x] 3.3 The Component's logit comparison with `transformers` (dense and shared-KV bundles) run against this change.

## 4. Validation

- [x] 4.1 `cargo fmt`, `clippy -D warnings`, `magnetar-runtime` tests, the wasm32 check.
