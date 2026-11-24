## Why

Issue #187 asks the Runtime to run a family of models whose layers differ from one another. The Model Component that builds their graphs (on its `d49fb7f` contract) uses only portable Operators. But it needs attributes and operand forms the catalog does not accept yet, and no Provider can run a graph the catalog rejects. This change adds them to the portable contract, the one every Provider implements. It also gives them reference semantics, which every Provider is checked against.

## What changes

- **Operator attributes (catalog):**
  - `attention.scale`: the factor applied to `QK^T`, replacing `1/sqrt(head_dimension)`.
  - `rope.partial_rotary_factor`: only `floor(factor * dimension / 2)` pairs rotate, and the rest keep frequency 0.
  - `gelu.approximate`: `"tanh"`, the portable `gelu`'s existing approximation.
  - `rmsnorm.head_count`, `rmsnorm.head_dimension`: per-head normalization with a `[head_dimension]` weight shared by every head.
  - `activation.cap` (`kind = "softcap"`: `cap * tanh(x / cap)`) and `activation.factor` (`kind = "scale"`: `x * factor`).
  - `embedding.scale`: the gathered rows multiplied by `scale`.
  - `moe-experts.activation`: `"silu"` (the default) or `"gelu_tanh"`, the experts' gate activation.
- **Operand forms (catalog):**
  - Trailing optional inputs: an Operator may accept fewer inputs than its maximum.
  - `rmsnorm` with no weight: one input, scale 1.
  - `moe-experts` with a sixth input, `per_expert_scale` (`[experts]`), multiplying each expert's output before the weighted sum.
  - `mul` with a second operand of shape `[1]`, broadcast to every element.
- **Reference semantics:**
  - Pure functions shared by every Provider go into `magnetar-provider-api`'s `operator_semantics`.
  - The Runtime's Reference CPU Provider implements every new form, and conformance fixtures pin each one against an independent computation.

Not in this change:
- WIT 1.6.0 (`layer-layout`, `options`);
- per-layer KV geometry and shared KV layers;
- the first-native engine's dispatch of the new forms;
- loaders;
- the CPU-optimized and CUDA Providers' Kernels.

Each is a later change, in the order given on issue #187.

## Impact

- **`magnetar-core`:** the operator catalog gains attribute rules, a minimum input count and a scalar-broadcast shape rule.
- **`magnetar-provider-api`:** `operator_semantics` gains the shared functions.
- **`magnetar-runtime`:** the Reference CPU kernels, dispatch and tests.

Existing graphs validate and compute exactly as before: every addition is optional, and absent attributes keep today's behavior.
