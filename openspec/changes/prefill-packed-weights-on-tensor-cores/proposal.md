## Why

Mistral 7B Q4_K_M prefills 4,096 tokens in about 39 s on an RTX 3060
(run 44256946982), roughly 1.5 TFLOP/s; Qwen2.5-0.5B Q8_0 takes 1.7 s. A
CUDA matmul against a packed weight reads it packed only for up to 64
rows. A longer prefill dequantizes the weight to f32 for the call and runs
the f32 CUDA-core Kernels, because the Tensor Core Kernels read weights as
bfloat16 and a dequantized packed weight has no bfloat16 copy. Matmuls are
about 93% of a 7B prefill's arithmetic at 4,096 tokens.

## What Changes

- **Split weight on Tensor Cores.** For more than 64 rows, a packed weight
  dequantized for the call is split, like the activations already are,
  into its bfloat16 hi and lo terms. `matmul_tcp_split_kernel` multiplies
  them on Tensor Cores with f32 accumulation: `a_hi·b_hi + a_lo·b_hi +
  a_hi·b_lo`, dropping only the product of both low terms, about 2^-18 of
  each product. Bias and residual addends are fused; a gated matmul over
  packed weights takes it through its two matmuls.
- Every other case is unchanged: up to 64 rows (packed row Kernels), no
  Tensor Cores (f32 Kernels), unaligned dimensions.

## Capabilities

### New Capabilities

- `packed-prefill-tensor-cores`: long matmuls against packed weights on
  Tensor Cores with split weights.

## Impact

- `providers/cuda`: `matmul_tcp_split_kernel`, the long packed matmul path,
  conformance.
- No Runtime, Component or WIT change.

## Non-goals

- Integer Tensor Core matmuls on the packed codes with quantized
  activations (no dequantization).
- Keeping a dequantized copy of the weights between calls.
