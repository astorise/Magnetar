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

## 6. Engine

- [x] 6.1 Head-wise `concat` admitted in Component graphs and dispatched; row `concat` refused there.
- [x] 6.2 `attention.value_head_dimension` forwarded with the output sized from it; YaRN forwarded on `rope`.
- [x] 6.3 V widths in the KV estimates (`kv_value_row_widths`, `layer_value_row_widths`).
- [x] 6.4 Split latent-attention projections in weight shapes, expected shapes and inventory; dense MLP outside `moe_layers`; optional shared expert gate; selection bias.
- [x] 6.5 Tests:
  - an expanded latent-attention layer (split projections, YaRN on both rotated parts, head-wise joins, a narrower V) prefills and decodes through the graph executor, history concatenated or in place, against a host oracle, with K and V stores at their own widths;
  - weight shapes and inventory for DeepSeek-V2-Lite's shape;
  - the V-width estimate;
  - the Reference CPU declares the new forms.

## 7. Loaders

- [x] 7.1 Hugging Face: latent attention from `kv_lora_rank`; YaRN from `rope_scaling` or a flat `rope_parameters`; routed experts, `moe_layers` and `expert_routing` from DeepSeek's keys.
- [x] 7.2 Hugging Face: row splits of `q_proj`/`q_b_proj`, `kv_a_proj_with_mqa` and `kv_b_proj`; experts stored one by one stacked; router, selection bias and shared experts named.
- [x] 7.3 GGUF `deepseek2`: the metadata of both converter forms, YaRN as llama.cpp applies it, routing and selection bias.
- [x] 7.4 GGUF `deepseek2`: row splits; `attn_v_b` as the value projection; `attn_k_b` transposed back per head, dequantized when block-quantized.
- [x] 7.5 Tests:
  - DeepSeek-V2-Lite, V2 and V3 configurations, a tiny transformers export and YaRN's `attention_factor` rules;
  - the split projections against transformers 5.23's own computations on tiny random bundles, with and without a low-rank query (exact; experts within 1e-10);
  - the real V2-Lite GGUF's metadata and tensor names; both key and value forms giving the same projections; a `Q8_0` `attn_k_b`; refusals of incomplete metadata, mismatched shapes and unknown gating;
  - a mutation of either split's row order is caught.

