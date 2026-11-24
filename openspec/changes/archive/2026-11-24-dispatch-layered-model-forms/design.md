## Decisions

### One feature per optional form, checked before dispatch

Each optional form a Kernel could silently ignore gets a Kernel feature. `required_kernel_features` maps a node to the features it needs. It lives in `magnetar-core`, next to the catalog, so every Runtime path and every Provider shares one list.

Before dispatch, the first-native engine requires each feature from every active Kernel of the bound Provider for that operator. Requiring all of them, rather than one, follows the banded-RoPE precedent: any of those Kernels may be selected. A missing feature refuses the dispatch with a Kernel-unavailable error.

Absent attributes and default values (`partial_rotary_factor = 1`, `activation = "silu"`) need no feature, so the graphs of uniform models never require one.

Forms that a Kernel already refuses explicitly need no feature:
- an unknown `activation` kind;
- a `mul` whose operands differ in shape, on a Kernel without broadcast.

### `with_scale` is accepted, and the input count wins

The layered-model Component marks its weightless RMSNorms with `with_scale = false`. The catalog accepts the attribute; the input count still decides. The Reference CPU refuses a node whose `with_scale` contradicts its input count, so the two can never disagree silently.

### `embedding` width from the table

A per-layer input table is narrower than the hidden size, so the output width is the table's row width.

## Evidence

`every_layered_operator_form_reaches_the_kernel` runs this graph on the Reference CPU:
- `embedding(scale)`;
- weightless per-head `rmsnorm`, then per-head `rmsnorm` with a `[head_dimension]` weight;
- partial half-split `rope`;
- `gelu(tanh)`;
- `mul` by a `[1]` operand;
- `activation(scale)`;
- `moe-experts` with `gelu_tanh` and `per_expert_scale`;
- a residual add of the scaled embeddings;
- the head;
- `activation(softcap)`.

The logits match a host computation within 1e-4. That computation is written by hand, except for the routed experts, which use the shared `operator_semantics` function.

Removing any one of these from the dispatch makes the test fail; this was checked by hand:
- the embedding's attributes;
- `partial_rotary_factor`;
- the sixth `moe-experts` input.

Two of these checks first passed while the forwarding was missing, and the test was then corrected:
- **Embedding scale:** at first the scaled embeddings fed only the RMSNorm, which cancels a scale. They now also feed a residual add.
- **Partial RoPE:** with base 10000, a non-rotating pair that rotates anyway moves too little to fail the comparison. The test now uses base 4.
