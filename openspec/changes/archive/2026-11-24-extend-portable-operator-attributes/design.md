## Context

The Model Component behind issue #187 emits portable Operators only; it never addresses a Provider. Provider-independent semantics live in two places:
- the operator catalog (`magnetar-core::operator`), which validates attributes, input counts and shapes;
- `magnetar-provider-api::operator_semantics`, whose pure functions Providers share so results and accumulation order agree everywhere.

The Runtime's Reference CPU Provider is the conformance oracle that the other Providers are compared with.

## Decisions

### Catalog

- **Optional trailing inputs.** `OperatorSpec` gains `min_inputs` (default: `inputs`). An invocation is valid with between `min_inputs` and `inputs` inputs. `rmsnorm` takes 1 or 2 inputs, and `moe-experts` 5 or 6.
- **Scalar broadcast.** `mul` takes the new `ShapeRule::SameShapeOrScalar`: the second operand matches the first, or has shape `[1]`. No attribute selects the broadcast: the shape does.
- **RMSNorm shapes.**
  - With `head_count = h` and `head_dimension = d`, the input's last dimension must be `h * d`, and the weight is `[d]`.
  - Without them, today's weight shapes apply.
  - The one-input form has no weight.
- **Attributes.**
  - Every new attribute is optional, and absent means today's behavior.
  - Values are checked by the kernels, not by the catalog:
    - `cap` and `factor` must be finite, and `cap` must be positive;
    - `partial_rotary_factor` must lie in `(0, 1]`;
    - `gelu.approximate` and `moe-experts.activation` must be one of the values listed above.

### Semantics

- **`attention.scale`.** It replaces `1/sqrt(head_dimension)` in the scores. Masking, windows and `key_count` are unchanged.
- **Partial RoPE.**
  - `rotated_pairs = floor(partial_rotary_factor * dimension / 2)`.
  - Pair `i` keeps frequency `base^(-2i / dimension)` for `i < rotated_pairs`; the remaining pairs have frequency 0 and stay unchanged.
  - The pairing (`interleaved` or `half_split`) still spans the whole `dimension`. The frequency denominator stays the full `dimension`, as the proportional scheme requires.
- **`gelu.approximate = "tanh"`.** The portable `gelu` is already the tanh approximation, so this attribute only states it. Other values are refused.
- **Per-head RMSNorm.** Each row splits into `head_count` blocks of `head_dimension`. Each block is normalized by its own mean square, then multiplied by the shared weight.
- **Scale-less RMSNorm.** It is `x * rsqrt(mean(x^2) + epsilon)`. It combines with the per-head form.
- **`activation`.**
  - `softcap` is `cap * tanh(x / cap)`.
  - `scale` is `x * factor`.
  - Both are elementwise and keep the shape.
- **`embedding.scale`.** The gathered row is multiplied by `scale`, after the gather.
- **`moe-experts`.**
  - With `activation = "gelu_tanh"`, each expert computes `gelu_tanh(gate) * up` instead of `silu(gate) * up`.
  - With the sixth input, expert `e`'s output is multiplied by `per_expert_scale[e]` before it is weighted by its routing weight and summed.
  - Routing (softmax, top-k, renormalization) is unchanged.
- **`mul` by a scalar.** Every element is multiplied by the second operand's one value.

### Where the code goes

- The pure functions go into `magnetar-provider-api::operator_semantics`:
  - softcap and scale;
  - per-head and scale-less RMSNorm;
  - scalar `mul`;
  - the MoE expert activation and per-expert scale.
- The Reference CPU Provider calls them, and other Providers may too.
- Attention and RoPE keep their Reference CPU implementations, which gain the new parameters.

## Risks / Trade-offs

- **The input-count change touches every `OperatorSpec` user.** The default keeps today's exact counts.
- **Only the Reference CPU implements the new forms.** Until a Provider advertises them, Kernel selection for a graph that uses them fails closed on other Providers. This is intended.

## Migration

Additive. No WIT change. Graphs and Kernels that use none of the new attributes or forms are unaffected.
