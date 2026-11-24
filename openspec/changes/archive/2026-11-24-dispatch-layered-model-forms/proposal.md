## Why

The Reference CPU computes every optional operator form the layered models of issue #187 need (`extend-portable-operator-attributes`). The first-native engine still could not run a graph using them:
- `embedding` dropped `scale` and sized its output at the architecture's hidden size;
- `rmsnorm` required two inputs and dropped the per-head attributes;
- `rope` dropped `partial_rotary_factor`;
- there was no `gelu` dispatch;
- `moe-experts` required five inputs.

Forwarding these forms also needs a guard. The CUDA Kernels ignore `embedding.scale`, the `rmsnorm` per-head attributes, `rope.partial_rotary_factor`, `moe-experts.activation` and a sixth `moe-experts` input, so they would compute something else without failing.

The Model Component for this family also marks its weightless RMSNorms with `with_scale = false`, an attribute the catalog did not accept.

## What changes

- **`magnetar-core`:**
  - `required_kernel_features(operator, attributes, input_count)` names the Kernel features an optional form needs:
    - `attention-scale`;
    - `embedding-scale`;
    - `rmsnorm-per-head`;
    - `rmsnorm-without-weight`;
    - `partial-rotary`;
    - `moe-experts-activation`;
    - `per-expert-scale`.
  - Baseline forms and default values need none.
  - `rmsnorm` accepts an optional boolean `with_scale`. The input count stays authoritative.
- **First-native dispatch:**
  - Before dispatching a node, the engine requires that every active Kernel of the Provider for that operator advertise each feature the node needs. This replaces the attention-only `scale` check.
  - `embedding` forwards its attributes and takes its output width from the table.
  - `rmsnorm` accepts one or two inputs and forwards `head_count`, `head_dimension` and `with_scale`.
  - `rope` forwards `partial_rotary_factor`.
  - `gelu` is dispatched, with its attributes.
  - `moe-experts` accepts a sixth input, `per_expert_scale`.
  - `activation` (softcap, scale) and the `[1]`-broadcast `mul` already reached the Kernel unchanged. CUDA refuses an unknown activation kind and a shape mismatch in `mul`.
- **Reference CPU:**
  - It advertises every feature above.
  - Its `rmsnorm` refuses a `with_scale` that contradicts its input count.

Not in this change:
- the CUDA Kernels for these forms;
- the logits-row gather after a softcap;
- the loaders.

## Impact

- **`magnetar-core`:** the operator catalog and its feature helpers.
- **`magnetar-runtime`:** the first-native dispatch, the Reference CPU, and tests.

Graphs that use none of these forms behave as before: they need no feature, and their dispatch is unchanged.
