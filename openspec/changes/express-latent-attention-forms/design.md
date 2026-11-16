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

- **Loaders split, the Runtime never fuses.** Both loaders cut the fused
  projections by rows into the projections the graph uses
  (`latent_attention_projection_shapes`); a row split copies bytes, so it is
  exact for float and block-quantized rows alike. A GGUF `attn_k_b` holds
  each head's key rows transposed, with its quantization blocks along the
  wrong dimension: its heads are transposed back, in `f32` when it is
  block-quantized.
- **Group score by scoring function in GGUF files.** A GGUF file does not
  record `topk_method`. Its group score follows the checkpoints' own routing:
  the best expert for softmax routers (DeepSeek-V2's
  `group_limited_greedy`) and the best two for sigmoid routers (V3's
  `noaux_tc`). llama.cpp sums the best two in both cases.

## Risks

- **YaRN conventions differ** between Hugging Face and GGUF files (llama.cpp
  derives `mscale` from `yarn_log_multiplier`). The Hugging Face loader keeps
  `attention_factor = mscale(factor, mscale) / mscale(factor,
  mscale_all_dim)`. A GGUF file does not record `mscale`, and llama.cpp
  leaves cosines and sines unscaled, so the GGUF loader declares
  `attention_factor = 1`. Both agree when `mscale` equals `mscale_all_dim`,
  as in the published DeepSeek-V2 and V3 configurations. `mscale_all_dim`
  from a GGUF file carries the `f32` rounding of `0.1 * m`.
- **The split GGUF form was not read from a converted file.** The
  `attn_k_b`/`attn_v_b` test writes them from `kv_b` as the converter's source
  does (`view`, `split`, `transpose(1, 2)`); running llama.cpp's converter
  was not possible in this environment.
