## Decisions

### One `score_scale` argument, with today's value as its default

Every attention Kernel takes a trailing `score_scale`, and a value `<= 0` keeps `1/sqrt(head_dimension)`. The Kernels covered are:
- generic and block;
- tiled;
- split and grouped decode;
- Tensor Core prefill, where the scale is folded into Q.

The host passes `0` when the node sets no scale, so the existing graphs' arithmetic is unchanged, bit for bit. A set scale must be finite and positive.

### Per-head RMSNorm reuses the row Kernel

The heads of a row are contiguous, so per-head normalization is the existing row Kernel over a `[rows * heads, head_dim]` view with a broadcast `[head_dim]` weight. A null weight is the scale-less form.

### Partial rotation in the RoPE Kernel

A pair past `floor(factor * dimension / 2)` gets angle 0, so the Kernel writes it unchanged. The Kernel still writes every element, so no input copy is needed and the step stays capturable.

### Per-expert scales after the renormalization

The routing Kernel multiplies each chosen expert's weight by its `per_expert_scale` entry after the renormalization, as `transformers`' Gemma 4 router does. This is equivalent to scaling that expert's output.

## Evidence

The CUDA conformance tests compare every form with the Reference CPU (`attention_scaled`, `rope_partial`) and with `operator_semantics`:
- **Attention:** shapes reaching every attention Kernel path, including 256- and 512-wide heads, with three scales.
- **Experts:** f32 and packed `Q8_0`.

A hardware test dispatches every form through the Kernel Registry and `CudaExecutor` with the node attributes the Runtime sends. The GPU runner of astorise/Magnetar-provider-CUDA#42 runs both.
