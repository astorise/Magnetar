## 1. Contract

- [x] 1.1 WIT `1.6.0`: `layer-spec`, `layer-layout`, `architecture-options` and `options` in `model-config`, identical to the Component's WIT.
- [x] 1.2 `ModelArchitectureConfig`: `LayerSpec`, `layer_layout`, `final_logit_softcapping`, `per_layer_input_size`, and validation.
- [x] 1.3 `ModelConfigCapability`: `layer-layout` and `options` encoders.
- [x] 1.4 Register `1.6.0` alongside `1.2.0` to `1.5.0`.
- [x] 1.5 Loaders: default the new fields.

## 2. Tests

- [x] 2.1 Capability encoding for uniform and layered configurations, and argument refusal.
- [x] 2.2 Layout validation refusals.
- [x] 2.3 A WAT Component importing `model-config@1.6.0` reads both functions through the canonical ABI.

## 3. Validation

- [x] 3.1 `cargo fmt`, `clippy -D warnings` and tests for `magnetar-model`, `magnetar-engine`, `magnetar-runtime` and both loaders.
