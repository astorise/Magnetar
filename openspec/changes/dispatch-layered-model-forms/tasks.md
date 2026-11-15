## 1. Catalog

- [x] 1.1 Feature constants and `required_kernel_features`; `rmsnorm.with_scale`.

## 2. Engine and Reference CPU

- [x] 2.1 Generic feature gate before dispatch, replacing the attention-only one.
- [x] 2.2 `embedding` attributes and table width; `rmsnorm` one or two inputs and per-head attributes; `rope` partial rotation; `gelu`; six-input `moe-experts`.
- [x] 2.3 Reference CPU advertises the features and checks `with_scale` against its inputs.

## 3. Tests

- [x] 3.1 A graph of every form against a host computation; a hand mutation check per forwarded form.
- [x] 3.2 `required_kernel_features` and the Reference CPU's declarations.

## 4. Validation

- [x] 4.1 `cargo fmt`, `clippy -D warnings`, and tests for `magnetar-core`, `magnetar-provider-api`, `magnetar-model`, `magnetar-engine` and `magnetar-runtime`.
