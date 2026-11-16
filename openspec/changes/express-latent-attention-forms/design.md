## Context

MLA (DeepSeek-V2/V3) computes, per layer:

- `q = [q_nope | rope(q_pe)]` per head;
- a latent `c_kv` and one shared rotated key `k_pe`;
- `k = [W_k c_kv | k_pe]` and `v = W_v c_kv` per head;
- attention with Q/K width `qk_nope + qk_rope` (192) and V width `v_head_dim`
  (128).

The loaders split each fused projection by rows (issue #198, decision 2), so
the graph needs no multi-output node and no slice. Two operations remain
that the catalog lacked: the head-wise join, and attention with distinct
widths.

## Decisions

- **`concat` axis rather than a new operator.** The graph executor has no
  `concat` arm for Component graphs (KV-history concatenation is
  Runtime-internal), so a head-wise `concat` cannot be mistaken for it.
  The feature gate keeps Providers that only join rows from receiving it.
- **An explicit `second_head_count`.** A width alone cannot say whether
  `[T, 64]` is one head of 64 or 16 heads of 4, so the broadcast is declared,
  never inferred. Only 1 and `head_count` are accepted.
- **Value width on `attention`.** The output becomes
  `[T, head_count * value_head_dimension]`. `scale` already replaces
  `1/sqrt(head_dimension)`, so YaRN's `mscale^2` needs no attribute of its
  own.
- **YaRN computed once.** `RopeYarnFrequencyScaling::correction_range`
  computes the range in `f64`, as transformers does in Python, and returns
  the `f32` bounds. `scale_frequency` blends each pair's frequency with
  `frequency / factor` along the clamped ramp. `attention_factor` multiplies
  `cos` and `sin`. Each parameter is required, so a Kernel never assumes one.
- **No routing attribute in this change.** The routed scaling factor is
  linear in the output; group-limited and sigmoid routing come in a later
  milestone with their own Kernel forms.

## Risks

- **YaRN conventions differ** between Hugging Face and GGUF files (llama.cpp
  derives `mscale` from `yarn_log_multiplier`). The loaders will map both onto
  these attributes; the golden test pins the Hugging Face definition.
