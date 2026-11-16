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
