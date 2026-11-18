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

Each value's split (round to nearest) loses at most 2^-18 of it, and the
dropped `a_lo·b_lo` is at most 2^-18 of the product: about 17 significant
bits per product, the precision of the Tensor Core path for compact
bfloat16 weights (whose weights are exact but whose activations are split
the same way). The weight's high term alone would lose up to 2^-9 of each
weight. This is less than the f32 CUDA-core path it replaces: a sum whose
products cancel keeps the products' absolute error (see Open question).

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

## Evidence

One RTX 3060 (12 GB), run 44266581438 against run 44256946982 (same
checkpoints, prompts and chunking, f32 long packed matmuls):

| TTFT | Before | Tensor Cores | Speedup |
|---|---|---|---|
| Mistral 7B Q4_K_M, 4,096 tokens | 40.01 s | 13.13 s | 3.05x |
| Mistral 7B Q4_K_M, 16,384 tokens | 170.47 s | 62.76 s | 2.72x |
| Mistral 7B Q4_K_M, 32,768 tokens (`tq4`) | 365.74 s | 154.14 s | 2.37x |
| Qwen2.5-0.5B Q8_0, 4,096 tokens | 1,688.6 ms | 803.7 ms | 2.10x |
| Qwen2.5-0.5B Q8_0, 16,384 tokens | 8,229.7 ms | 4,640.2 ms | 1.77x |
| Qwen2.5-0.5B Q8_0, 32,768 tokens | 20,379.5 ms | 13,075.6 ms | 1.56x |

- The generated tokens after the chunked prefill test's prompts are the
  same as with the f32 path (Qwen2.5-0.5B 3,000 tokens, Mistral 7B 4,000,
  16 tokens each); chunked and one-pass prefills still give bit-identical
  first-step logits. TPOT is unchanged.
- `long_prefills_against_packed_weights_match_reference_cpu` passes on the
  GPU.
- `q8_packed_gated_matmul_and_embedding_match_reference_cpu` fails at 70
  rows: 150.4519 against 150.4536, beyond its f32 tolerance
  (`1e-5 + 1e-5 · k`, 1.29e-3 for k = 128).

## Open question

The failing test's 70-row case now runs on this path. Simulated with exact
accumulation, two bfloat16 terms per operand exceed its tolerance with 3
or 4 products (largest difference 6.6e-3 and 3.9e-3, outputs up to about
700); three terms per operand with the 6 products down to 2^-24
(`hi·hi, hi·mid, mid·hi, hi·lo, mid·mid, lo·hi`) stay within it (3.7e-4).
Either the long packed path keeps f32 precision with 6 products (twice the
Tensor Core work of 3), or it takes the compact-weight Tensor Core
precision and that test case its Tensor Core tolerance. Pending decision.

