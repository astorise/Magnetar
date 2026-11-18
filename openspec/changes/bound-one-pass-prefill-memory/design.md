## Context

A prefill that cannot run in chunks holds every prompt row's activations in
every node. Intermediates are released after their last consumer, so its
peak is a few nodes' outputs at once; but some nodes held more than their
output.

## Decisions

### 1. The logits row gather on every prefill graph

Every prefill samples one row: a bucket's (its last real row), a chunked
prefill's passes (each its last real row), and a prompt run in one pass at
its own length (its last row). `ComponentGraphCache::prepare` now inserts
the gather (`insert_logits_row_gather`) in every prefill graph outside
tensor parallelism, not only in a bucket's and the chunk's; the engine
already feeds `input.logits_row` to any graph that takes it. The production
loop checked `logits_row` to tell a bucket's or the chunk's graph (whose
padding needs KV edges read by attention alone); it now asks
`is_prefill_bucket` directly, or a one-pass graph would never qualify.

### 2. Row blocks in the split Tensor Core matmul

`matmul_tcp_split3_kernel` multiplies three bfloat16 terms of A and of B.
The terms are kept in a split buffer that grows to the largest call and is
kept (a captured step may read it). B's terms are now split once, and A's
2,048 rows at a time into one block's room, each block's product written at
its rows of the output (a full addend's rows follow). A row's value depends
only on its own terms and B's, so it is unchanged; a call of at most 2,048
rows (every captured one) launches as before.

### 3. A gated MLP over packed weights in row blocks

With more rows than the packed Kernels take, `gated_matmul` ran both
projections' packed matmuls (each expanding its weight) and `silu_mul`.
`gated_matmul_split_tcp` expands both weights once, splits them once, and
for each block of 2,048 rows writes the gate's and up's products into two
block-sized buffers, then `silu_mul_kernel` into the block's output rows.

### 4. A `moe-experts` prefill grouped by expert

The per-pair Kernels give each (token, expert) pair a block row and read
the expert's weights for it. For more tokens than a decode step's (64), and
packed `Q8_0` experts, the pairs are grouped by expert on the Device (a
count, one thread's offsets, a scatter) and each block serves up to eight
pairs of one expert, reading its weights once for them. The per-pair sums
of the down projection go to a buffer and `moe_down_combine_kernel` adds
them per token in routing order, as the per-pair down Kernel does. Every
value keeps the per-pair Kernels' lanes, order and reductions: the results
are theirs bit for bit, whatever the order of an expert's pairs. Tokens are
served 1,024 at a time, which bounds the activations and per-pair sums.
The per-pair prefill also launched one block row per pair: 98,304 pairs at
16,384 tokens and top 6 exceed a grid's 65,535 rows.

### 5. Tiled prefill attention with narrower values

`attention_tiled<PART, VPART>`: a query's four threads own `PART` key and
`VPART` value dimensions; V tiles and the output take the value width.
`attention_tiled_48_32_kernel` serves 192-dimension keys with
128-dimension values, which fell to the block Kernel (one block per head
and query, every key read from global memory) or, past its shared memory,
to one thread per head and query. Equal widths keep their arithmetic.

## Evidence

Run 44374637064 (providers/cuda 758b8ba, the layers balanced, before
decisions 1 to 4): DeepSeek-V2-Lite-Chat Q8_0 prefills 4,096 tokens (TTFT
24,776.6 ms in `f32`) where the filled placement ran out of memory; TTFT at
1,024 tokens is 5,989.3 against 6,975.8 ms with the untiled attention and the
first GPU filled (run 44358602056), the two changes not separated. At 16,384
tokens every precision runs out of memory in layer
0's `gate_proj` (the fused gated MLP), 3,890 MiB free on the first GPU
after the load.

Run 44375603855 (providers/cuda 90d5efd, the grouped expert prefill, eight
pairs and one column per warp): `grouped_moe_prefill_matches_the_per_pair_kernels`
passes (bit for bit). `moe_prefill_kernel_times`, DeepSeek-V2-Lite's experts
(2048 hidden, 1408 intermediate, 64 experts, top 6), packed `Q8_0`:

| Tokens | One pair at a time | Grouped by expert |
|---|---|---|
| 1,024 | 187.09 ms | 107.56 ms |
| 4,096 | 750.54 ms | 429.70 ms |

DeepSeek-V2-Lite-Chat's TTFT then falls from 5,989.3 to 3,800.4 ms at 1,024
tokens and from 24,776.6 to 16,281.5 ms at 4,096 (`f32` history), its 64
tokens and first-step logits still equal (16,384 tokens still run out of
memory: decisions 1 to 3 were not in that run). Each lane of that form
converts its quants again for every pair and reads its values again for
every column; providers/cuda 49a84d2 serves two columns per warp, each
lane converting its quants once per step and reading a pair's values once
for both columns.

## Risks

- **Blocks re-split A.** A's split is per block, as before per call; B's
  once per call. No extra work.
- **The gather everywhere.** A graph whose LM head cannot take the gather
  keeps every row's logits, as a bucket's did.
