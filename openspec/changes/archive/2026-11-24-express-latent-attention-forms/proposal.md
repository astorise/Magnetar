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
- `model-config` `1.7.0` (the review's decisions 1, 6, 7, 8 and 9):
  - `multi-head-latent-attention: func() -> option<mla-config>`;
  - `mixture-of-experts-layers: func() -> option<list<bool>>`;
  - `expert-routing: func() -> option<expert-routing-config>`, with
    `expert-scoring`, `expert-groups` and `expert-group-score`;
  - a `yarn` case of `rope-frequency-scaling` (`yarn-rope-scaling`).

  The record is named `expert-routing-config` because WIT forbids a type and
  a function sharing a name. `ModelArchitectureConfig` and
  `FirstNativeModelConfig` carry the matching fields, and
  `ModelArchitectureConfig::validate` checks them against the architecture.
  The host serves `1.2.0` to `1.7.0`.
- **Fail closed on unread features.** The model-config capability records the
  functions each instance calls. The Runtime refuses a graph built while the
  configuration declares latent attention, mixed expert layers or expert
  routing that the Component never read. YaRN is checked on the graph itself,
  like the banded scaling: every `rope` node must declare it.
- **Engine.**
  - Component graphs may use head-wise `concat`; its dispatch has its own
    branch. A graph's row `concat` is refused, since KV-history concatenation
    stays the Runtime's own.
  - `attention` forwards `value_head_dimension` and sizes its output from it.
  - `rope` forwards YaRN.
  - KV stores take each edge's own width, so expanded K and V stores already
    differ. The estimates now count V at `kv_value_row_widths`.
  - Weight shapes and inventory validation derive the split latent-attention
    projections (`latent_attention_projection_shapes`) instead of q/k/v/o, a
    dense MLP for layers outside `moe_layers`, no shared expert gate when
    routing says it is ungated, and a selection bias when routing declares
    one.
- **Loaders.** The Hugging Face loader reads DeepSeek-V2/V3 `config.json`
  files and the GGUF loader reads `deepseek2` files into these fields. Both
  split the fused latent-attention projections by rows into the projections
  the graph uses; the GGUF loader also reads the split `attn_k_b`/`attn_v_b`
  form of newer files.
- **CUDA Provider.** YaRN in the `rope` Kernel; `value_head_dimension` in the
  general attention Kernels (block, one thread per head and query,
  split-key decode), the Tensor Core, tiled and grouped decode Kernels
  keeping equal widths; a head-wise `concat` Kernel. It advertises the three
  features. Providers that do not advertise a feature never receive it; the
  CPU Provider crate does not implement these forms, as it does not
  implement the layered forms, and the Runtime's Reference CPU serves them.
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
- `magnetar-model`: the `1.7.0` types and fields, and their validation.
- `magnetar-engine`: `model_config_capability.rs` (encoding, call tracking).
- `magnetar-runtime`: `wit/model-component-graph.wit` (`1.7.0`),
  `component_graphs.rs` (versions, unread-feature refusal), `engine.rs` (YaRN
  on `rope` nodes), and the config conversions.
- `loaders/huggingface`: `config.rs` (latent attention, YaRN, expert layers
  and routing), `naming.rs`, `layered_tensors.rs` (row splits, expert
  stacking).
- `loaders/gguf`: `deepseek2` in `config.rs`, `naming.rs` and
  `layered_tensors.rs` (row splits, the `attn_k_b` head transpose).
- `providers/cuda`: `kernels.cu` (YaRN in `rope_kernel`; the value width in
  `attention_kernel`, `attention_block_kernel` and `attention_split_kernel`;
  `concat_heads_kernel`), `kernels.rs` (`rope_scaled`, `attention_valued`,
  `concat_heads`), `executor.rs`, `advertisements.rs`, conformance tests.
- Existing graphs and Kernels are unchanged: every new attribute is optional,
  and a Provider receives one only if it advertises the feature. Existing
  Components keep linking.
