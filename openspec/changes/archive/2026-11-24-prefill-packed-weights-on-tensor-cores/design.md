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
product.

### 3. A partial per tile, added in f32

The Tensor Cores' f32 accumulation aligns the products and the
accumulator to the largest of them and truncates the rest. Accumulated in
one chain over every tile, the largest term is the running sum, and the
truncations of every `mma` add up with one sign. The six products of
each 16-inner tile therefore accumulate into a partial that starts at
zero, and an ordinary f32 add (round to nearest) takes the partial into
the output: the result is as precise as the f32 Kernels'.

With six products in one chain, the gated packed `Q8_0` conformance case
at 70 rows exceeded its f32 tolerance on the RTX 3060 (-404.9817 against
-404.9833, run 44274927793). Modelled with truncating accumulation, the
chain gives 1.5e-3 (as measured) and the per-tile partials 3.7e-4, with or
without guard bits, against 1.29e-3 allowed.

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

### 4. Where

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

One RTX 3060 (12 GB), the same checkpoints, prompts and chunking (2,048-row
passes) in every run. Run 44256946982 used the f32 long packed matmuls;
run 44266581438 the first version (two terms, three products); run
44283381146 the kept version (three terms, six products, a partial per
tile added in f32):

| TTFT | f32 | Two terms, 3 products | Three terms, 6 products | Kept vs f32 |
|---|---|---|---|---|
| Mistral 7B Q4_K_M, 4,096 tokens | 40.01 s | 13.13 s | 22.85 s | 1.75x |
| Mistral 7B Q4_K_M, 16,384 tokens | 170.47 s | 62.76 s | 102.02 s | 1.67x |
| Mistral 7B Q4_K_M, 32,768 tokens (`tq4`) | 365.74 s | 154.14 s | 232.69 s | 1.57x |
| Qwen2.5-0.5B Q8_0, 4,096 tokens | 1,688.6 ms | 803.7 ms | 1,301.5 ms | 1.30x |
| Qwen2.5-0.5B Q8_0, 16,384 tokens | 8,229.7 ms | 4,640.2 ms | 6,639.9 ms | 1.24x |
| Qwen2.5-0.5B Q8_0, 32,768 tokens | 20,379.5 ms | 13,075.6 ms | 17,090.8 ms | 1.19x |

Every CUDA conformance test passed at its unchanged tolerance in run
44283381146, the gated packed `Q8_0` case at 70 rows included. The six
products in one chain (run 44274927793, which failed that case) took the
same time to within 1.5% (Mistral 7B 22.5 s and 100.6 s at 4,096 and
16,384 tokens): the per-tile add costs nothing measurable. The kept
version takes 1.3x to 1.7x the first version's time: twice the Tensor Core
work, and three split terms per operand instead of two.

The tokens generated after the chunked prefill test's prompts (Mistral 7B,
4,000 tokens; Qwen2.5-0.5B, 3,000) were the same in all four runs, and
chunked and one-pass prefills stayed bit-identical (run 44283381146:
largest first-step logit difference 0 for chunks of 2,048 and 1,024 rows,
and of 1,024 and 512).
