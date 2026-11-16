## 1. Catalog and shared semantics

- [x] 1.1 `attention.value_head_dimension`, `concat` axis/head counts, YaRN `rope` attributes; Kernel features; head-wise `concat` shape rule.
- [x] 1.2 `RopeYarnFrequencyScaling` (correction range, frequency blend, attributes) and `concat_heads_widths` in `magnetar-core`; `concat_heads` in `operator_semantics`.

## 2. Reference CPU

- [x] 2.1 `attention_with_value_width`; `rope_scaled` with a per-pair frequency table and YaRN's amplitude; the `concat` axis.
- [x] 2.2 Advertise `attention-value-head-dimension`, `concat-heads` and `yarn-frequency-scaling`.

## 3. Tests

- [x] 3.1 Attention with a narrower V against a direct computation; refusal of mismatched V rows.
- [x] 3.2 Head-wise `concat`, per head and broadcast; refusal of ambiguous widths and unknown axes; row `concat` unchanged.
- [x] 3.3 YaRN frequencies against transformers' `_compute_yarn_parameters` (two configurations); `attention_factor`; a missing parameter refused; a mutation of the ramp is caught.
- [x] 3.4 Kernel features and the shape rule in `magnetar-core`.

## 4. Validation

- [x] 4.1 `cargo fmt`, `clippy -D warnings`, rustdoc and tests for `magnetar-core`, `magnetar-provider-api` and `magnetar-runtime`; `providers/cpu` and `providers/cuda` still compile.

## 5. Contract `1.7.0`

- [x] 5.1 WIT: `mla-config`, `yarn-rope-scaling` and the `yarn` case, `mixture-of-experts-layers`, `expert-routing-config` with its enums and groups; validated by `wasm-tools`.
- [x] 5.2 `ModelArchitectureConfig`/`FirstNativeModelConfig` fields and their validation; the host encoding; `1.7.0` linked beside `1.2.0`–`1.6.0`.
- [x] 5.3 Call tracking and the refusal of graphs that leave a declared feature unread; YaRN on every `rope` node.
- [x] 5.4 Tests:
  - a `1.7.0` probe Component reading all four functions across the real component boundary;
  - a `1.4.0` Component failing to receive YaRN;
  - the real Qwen Component refused for mixed layers, routing and YaRN;
  - validation of inconsistent declarations.
