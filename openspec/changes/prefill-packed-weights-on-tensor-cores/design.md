## Context

`CudaKernels::matmul_with_addend` reads a packed weight (`Q8_0`, `Q4_K`,
`Q5_K`, `Q6_K`, `Q2_K`, `Q3_K`, the i-quants, `Q4_0`) with its row Kernels
for up to `Q8_MATMUL_MAX_ROWS` (64) rows. Above that it dequantizes the
weight to an f32 buffer and calls itself on it. The f32 buffer has no
bfloat16 address, so the Tensor Core Kernels (`matmul_tcp_kernel`, which
take the activations as two bfloat16 terms and a bfloat16 weight) do not
apply, and the tiled f32 Kernel runs.

## Decisions

### 1. Three bfloat16 terms per operand

`tc_split3_kernel` splits each f32 value into `hi = bf16(x)`, `mid =
bf16(x - hi)` and `lo = bf16(x - hi - mid)` (round to nearest; both
subtractions are exact in f32): about 24 significant bits, an f32 value's.
The dequantized weight and the activations are split in one call into the
Tensor Core split scratch (three halves per value).

### 2. Six products

`matmul_tcp_split3_kernel` keeps `tcp_matmul`'s tiles (128 x 64 outputs per
block, 16 inners per stage, two `cp.async` stages, `ldmatrix`, `m16n8k16`
bfloat16 `mma` with f32 accumulation). Its six terms' two stages take
50,688 bytes of dynamic shared memory. Per stage it accumulates every
product of terms down to 2^-24 of the whole, smallest first:

`lo·hi + mid·mid + hi·lo + mid·hi + hi·mid + hi·hi`.

The dropped products (`mid·lo`, `lo·mid`, `lo·lo`) are below 2^-25 of each
product: the result is as precise as the f32 Kernels'.

A first version split each operand into two terms and kept three products
(`a_hi·b_hi + a_lo·b_hi + a_hi·b_lo`, about 17 significant bits, the
compact-weight Tensor Core path's precision). On the RTX 3060 it exceeded
the f32 tolerance of an existing conformance case, a gated matmul over
packed `Q8_0` weights at 70 rows whose products cancel (150.4519 against
150.4536, run 44266581438). Simulated with exact accumulation, two terms
exceed that tolerance with three or four products (largest difference
6.6e-3 and 3.9e-3 against 1.29e-3 allowed), three terms with five; three
terms with these six stay within it (3.7e-4). The tolerance is kept; the
six products cost twice the Tensor Core work of three.

### 3. Where

The packed branch of `matmul_with_addend`, after the row Kernels: more
than 64 rows (`MATMUL_TILE_MIN_ROWS`), no transposition, both inner and
output dimensions multiples of 8. Otherwise the f32 path as before.

## Risks

- **Memory.** The call holds the dequantized f32 weight and its three terms
  in the split scratch (10 bytes per weight value, about 590 MB for a
  Mistral 7B MLP projection), as long as the matmul runs. The split scratch
  keeps its largest size.
- **Bandwidth.** The weight is dequantized and split at every call, once
  per prefill pass.

## Evidence

One RTX 3060 (12 GB). Run 44266581438 measured the first version (two terms,
three products) against run 44256946982 (same checkpoints, prompts and
chunking, f32 long packed matmuls):

| TTFT | f32 | Two terms, 3 products | Speedup |
|---|---|---|---|
| Mistral 7B Q4_K_M, 4,096 tokens | 40.01 s | 13.13 s | 3.05x |
| Mistral 7B Q4_K_M, 16,384 tokens | 170.47 s | 62.76 s | 2.72x |
| Mistral 7B Q4_K_M, 32,768 tokens (`tq4`) | 365.74 s | 154.14 s | 2.37x |
| Qwen2.5-0.5B Q8_0, 4,096 tokens | 1,688.6 ms | 803.7 ms | 2.10x |
| Qwen2.5-0.5B Q8_0, 16,384 tokens | 8,229.7 ms | 4,640.2 ms | 1.77x |
| Qwen2.5-0.5B Q8_0, 32,768 tokens | 20,379.5 ms | 13,075.6 ms | 1.56x |

The generated tokens after the chunked prefill test's prompts were the same
as with the f32 path, and chunked and one-pass prefills stayed bit-identical.
The three-term, six-product version is to be measured (task 2.1).
