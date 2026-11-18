## Context

`CudaKernels::matmul_with_addend` reads a packed weight (`Q8_0`, `Q4_K`,
`Q5_K`, `Q6_K`, `Q2_K`, `Q3_K`, the i-quants, `Q4_0`) with its row Kernels
for up to `Q8_MATMUL_MAX_ROWS` (64) rows. Above that it dequantizes the
weight to an f32 buffer and calls itself on it. The f32 buffer has no
bfloat16 address, so the Tensor Core Kernels (`matmul_tcp_kernel`, which
take the activations as two bfloat16 terms and a bfloat16 weight) do not
apply, and the tiled f32 Kernel runs.

## Decisions

### 1. Split the weight like the activations

`split_bf16_all` already splits f32 values into bfloat16 hi and lo terms
(`hi = bf16(x)`, `lo = bf16(x - hi)`) for the activations and for
attention's K and V. The dequantized weight is split the same way, into
the same scratch buffer, in one call with the activations.

### 2. Three products

`tcp_matmul<NB, LO_LAST>` is the pipelined Kernel's body over `NB` weights;
with `LO_LAST`, the last weight is the low term of the one before and is
multiplied by the activations' high term only. `matmul_tcp_split_kernel`
runs it over `{b_hi, b_lo}` and adds both accumulators:

`a_hi·b_hi + a_lo·b_hi + a_hi·b_lo`.

Each value's split loses at most 2^-17 of it, and the dropped `a_lo·b_lo`
is at most 2^-18 of the product: the result is f32-accurate to about 2^-16
per product, like the Tensor Core path for compact bfloat16 weights. The
weight's high term alone would lose up to 2^-9 of each weight.

### 3. Where

The packed branch of `matmul_with_addend`, after the row Kernels: more
than 64 rows (`MATMUL_TILE_MIN_ROWS`), no transposition, both inner and
output dimensions multiples of 8. Otherwise the f32 path as before.

## Risks

- **Memory.** The call holds the dequantized f32 weight and both terms of
  it in the split scratch (8 bytes per weight value, 470 MB for a Mistral 7B
  MLP projection), as long as the matmul runs. The split scratch keeps its
  largest size.
- **Bandwidth.** The weight is dequantized and split at every call, once
  per prefill pass.
