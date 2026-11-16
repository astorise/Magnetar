## Why

Issue #198: `components/deepseek` cannot build DeepSeek-V2/V3 graphs, because
the graph contract cannot express multi-head latent attention (MLA). The
Runtime review on the issue settled the `1.7.0` contract. Its first
milestone caches the expanded K and V and needs three new operator forms:

- per-head query and key rows wider than the value rows (192 and 128 for
  DeepSeek-V2), so attention must take a V width of its own;
- queries and keys assembled head by head from a non-rotated part and a
  rotated part, the key's rotated part being one head shared by every head;
- YaRN RoPE frequencies, which DeepSeek-V2 and V3 declare
  (`rope_scaling.type = "yarn"`).

## What Changes

- `attention.value_head_dimension` (optional): V rows and the output are that
  wide per head, while Q and K rows stay `head_dimension` wide. Kernel
  feature `attention-value-head-dimension`.
- `concat` gains `axis = "heads"`, `head_count` and `second_head_count`. It
  joins `[T, H * a]` and `[T, H * b]` (or `[T, b]`, one head given to every
  head) into `[T, H * (a + b)]`, head by head. The shape rule validates the
  widths. Kernel feature `concat-heads`. Without `axis`, `concat` still joins
  rows.
- `rope.frequency_scaling = "yarn"`, with `frequency_scaling_factor`,
  `original_context_length`, `beta_fast`, `beta_slow`, `attention_factor` and
  `truncate`, all required. `RopeYarnFrequencyScaling` in `magnetar-core`
  computes the correction range and each pair's frequency, as every Provider
  will. Kernel feature `yarn-frequency-scaling`.
- The Reference CPU implements and advertises the three forms.
- No `routed_scaling_factor` attribute: scaling every routing weight by `s`
  scales the routed experts' output by `s`. A Component expresses it with
  `activation{kind = "scale"}` before adding the shared expert.

## Impact

- `magnetar-core`: `operator.rs` (features, schemas, shape rule,
  `RopeYarnFrequencyScaling`, `ConcatAxis`).
- `magnetar-provider-api`: `operator_semantics::concat_heads`.
- `magnetar-runtime`: `reference_cpu.rs`
  (`attention_with_value_width`, `rope_scaled` with `RopeFrequencies`, the
  `concat` axis).
- Existing graphs and Kernels are unchanged: every new attribute is optional,
  and a Provider receives one only if it advertises the feature.
