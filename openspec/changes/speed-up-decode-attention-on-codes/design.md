## Context

`tq_attention_split_grouped_kernel` runs one block of 128 threads per
(chunk of 128 keys, KV head). It rotates every query head of the group
into shared memory (a static `TQ_MAX_GROUP x TQ_MAX_DIM` array, 16 KB),
scores the chunk's keys one thread per key, each thread reading its own
key's codes from global memory (rows `kv_heads x code_bytes` apart), and
sums the values one thread per value coordinate, each walking all 128
keys' codes in global memory. Every block writes a partial per query
head; `attention_combine_kernel` merges `ceil(rows / 128)` partials per
head and `tq_rotate_back_kernel` rotates the result back.

Measured (proposal), the Kernel reads 14 to 34 GB/s where the f32 grouped
Kernel, which stages its chunk's K and V rows in shared memory with
coalesced loads before working on them, reads 95 to 195 GB/s.

## Decisions

### 1. Stage the chunk's codes

The block first loads its chunk's K and V codes (`code_bytes` per row:
32 or 64 for `tq4`, 24 or 48 for `tq3`) and both norms into shared
memory, four bytes per thread per load, consecutive threads reading
consecutive words of a row; rows are padded by one word so a thread per
key reads distinct banks. Scoring and the value sums then read shared
memory only. An exact head's keys and values (`keep-turboquant-kv-head-
exact`) stay read from global memory: they occur in at most the first
`E / 128` chunks.

### 2. Every thread in the value sums

With `v_dim < 128`, thread `t` sums coordinate `t % v_dim` over the keys
`j` with `j % (128 / v_dim) == t / v_dim`; the slices are added in shared
memory in a fixed order. With 128-wide values, one coordinate per thread
as today.

### 3. Queries' shared memory sized to the group

The rotated queries take `group x head_dim` floats of dynamic shared
memory instead of a static 16 KB, so a multiprocessor holds more blocks
(about 4 today).

### 4. Split-K

Block `(s, kv_head)` of `S` blocks per KV head takes chunks
`[s * n / S, (s + 1) * n / S)` of the `n` chunks of admitted keys, `n`
computed on the Device from the key count (so a captured step's replays
spread their longer histories over the same blocks). It rotates the
queries once, walks its chunks with an online softmax (running maximum
and sum per query head, accumulators rescaled when the maximum grows),
and writes one partial per query head; an exact head's unrotated sums run
alongside and are rotated once at the end. The combine merges `S`
partials per head.

`S` is chosen on the host, after the RFC's heuristic:
`min(ceil(4 x SMs / kv_heads), max(1, chunks / 2), 32)` for the history
being decoded (at least two chunks per block, at most four blocks per
multiprocessor), then reduced to fill whole waves of the multiprocessors
when the last wave would be less than three quarters full. The blocks a
multiprocessor holds come from the Kernel's occupancy at its shared
memory size (at most four).

`S` grows with the history, and a captured step cannot grow the
attention scratch: the scratch is sized for `min(chunks, 32)` partials
per query head, `chunks` counted over the stores' capacity.

## Risks

- Shared memory for staged codes: 128 keys x (2 x 64 + 8) bytes, about
  17 KB for 128-wide `tq4` heads; with decision 3 the total stays below
  today's.
- Summation order changes: results move within f32 rounding; the
  conformance tolerances are unchanged.

## Evidence

One RTX 3060. Run 44299614540 (providers/cuda 34e39e6): the TurboQuant
conformance tests pass under the three schedules (unstaged, staged with a
chunk per block, staged and split), with 2,000- and 4,100-key histories,
and a captured split step replayed up to 1,001 keys matches the CPU.
`decode_attention_times`, one decode step's attention:

| `tq4`, us | Unstaged | Staged, chunk per block | Staged and split |
|---|---|---|---|
| Qwen2.5-0.5B, 4,096 keys | 41.3 | 45.3 | 57.9 |
| Qwen2.5-0.5B, 16,384 keys | 137.3 | 128.2 | 133.9 |
| Qwen2.5-0.5B, 32,768 keys | 225.5 | 215.7 | 232.6 |
| Mistral 7B, 4,096 keys | 154.7 | 170.5 | 169.8 |
| Mistral 7B, 16,384 keys | 515.1 | 536.2 | 491.3 |
| Mistral 7B, 32,768 keys | 1,015.8 | 1,037.8 | 971.5 |

`tq3` behaves alike (Mistral 7B at 32,768 keys: 1,113.3, 983.4 and
947.6 us). Neither staging nor the split changes the Kernel's speed by
more than about 10% either way, and the split is slower up to 4,096 keys,
where fewer blocks each walk their chunks one after another. The
unstaged Kernels stay the default.

What bounds the Kernels instead: at 32,768 keys the f32 grouped Kernel and
the code Kernels cost about the same per (key, KV head) pair (Qwen2.5-0.5B
3.9 and 3.4 ns, Mistral 7B 5.3 and 3.9 ns), though the codes are 7 to 8
times fewer bytes. Both do the same multiply-adds per pair (group x head
width, for the scores and for the values), and both read about one
shared-memory operand per multiply-add (a query or score value, a K or V
value or centroid). A suspected bound, shared to both: shared-memory loads
per multiply-add. The next step tests it with Kernels that keep the
queries and the weights in registers across several keys and coordinates
(decision 5).

### 5. Register tiling (next)

A thread scores several keys per query value it loads, and sums several
value coordinates per weight it loads (weights stored key-major, the
group's heads contiguous), so a shared-memory load feeds several
multiply-adds. First on the f32 grouped decode Kernel, whose bytes the
gain would show directly, then on the code Kernels.

Run 44302848278 (providers/cuda f404ebf) tiles the f32 grouped Kernel
(a thread scores a key against four or two heads from float4 loads, and
sums a value coordinate of four or eight heads from float4 weight loads;
results bit-identical). One decode step's attention:

| f32, us | 4,096 keys | 16,384 | 32,768 |
|---|---|---|---|
| Qwen2.5-0.5B, before | 44.4 | 136.5 | 252.9 |
| Qwen2.5-0.5B, tiled | 39.6 | 120.0 | 224.0 |
| Mistral 7B, before | 186.4 | 689.1 | 1,387.2 |
| Mistral 7B, tiled | 185.4 | 687.0 | 1,386.6 |

Qwen2.5-0.5B's attention (64-wide heads, 7 query heads per KV head) is
11% to 12% faster, and its f32 TPOT drops from 9.35 to 8.66 ms at 32,768
tokens and from 6.57 to 6.16 ms at 16,384 (run 44302848278 against
44289082020). Mistral 7B's (128-wide heads, 4 per KV head, already at
194 GB/s) does not move. Shared-memory loads per multiply-add bound part
of the 64-wide case only; they are not the whole bound.

Run 44312374879 (providers/cuda de9dffc) tiles the grouped code Kernel,
the default: weights key-major and read as float4s (four heads per
load), rotated queries read as float4s, and 64-wide value heads spread
over every thread (four heads per thread); results bit-identical, every
conformance test passing. One decode step's attention:

| us | 4,096 keys | 16,384 | 32,768 |
|---|---|---|---|
| Qwen2.5-0.5B `tq4`, before / tiled | 41.5 / 32.5 | 137.6 / 107.3 | 235.7 / 183.6 |
| Qwen2.5-0.5B `tq3`, before / tiled | 41.5 / 33.6 | 106.0 / 89.8 | 229.9 / 180.3 |
| Mistral 7B `tq4`, before / tiled | 154.7 / 108.1 | 512.0 / 357.6 | 987.3 / 666.4 |
| Mistral 7B `tq3`, before / tiled | 170.4 / 126.8 | 563.7 / 417.1 | 1,127.8 / 802.2 |
| f32 (tiled), Qwen2.5-0.5B / Mistral 7B | 39.7 / 185.2 | 120.2 / 686.0 | 224.0 / 1,386.2 |

The code Kernel is 22% faster for Qwen2.5-0.5B and 29% to 33% for
Mistral 7B; it now takes 0.82x and 0.48x f32's time at 32,768 keys. It
still reads 26 and 54 GB/s of the bytes it must, against 150 and
194 GB/s for f32: the next bound to find.

TPOT with both tiled Kernels (run 44314151729) against the f32-tiled run
before the code Kernel's tiling (44302848278); greedy decoding of 64
tokens on one RTX 3060:

| TPOT, ms | 4,096 | 16,384 | 32,768 |
|---|---|---|---|
| Mistral 7B f32 | 23.27 | 41.01 | does not fit |
| Mistral 7B `tq4` (head), before / tiled | 23.11 / 22.02 | 35.51 / 29.99 | 51.85 / 41.26 |
| Mistral 7B `tq4` (no head), before / tiled | 22.37 / 20.82 | 34.66 / 29.46 | 50.92 / 40.70 |
| Mistral 7B `tq3` (head), before / tiled | 23.53 / 22.66 | 37.62 / 32.50 | 56.04 / 45.88 |
| Qwen2.5-0.5B f32 | 4.24 | 6.15 | 8.68 |
| Qwen2.5-0.5B `tq4` (head), before / tiled | 4.74 / 4.74 | 6.74 / 6.01 | 9.00 / 7.89 |
| Qwen2.5-0.5B `tq4` (no head), before / tiled | 4.64 / 4.38 | 6.62 / 5.95 | 8.99 / 7.80 |
| Qwen2.5-0.5B `tq3` (head), before / tiled | 4.76 / 4.66 | 6.55 / 6.04 | 9.15 / 8.02 |

Mistral 7B now decodes 1.37x faster than f32 at 16,384 tokens in `tq4`
(1.26x in `tq3`), and 20% faster than before at 32,768; Qwen2.5-0.5B
1.10x faster than f32 at 32,768 tokens. The generated tokens and the
first step's logits are those of the run before (the Kernels' sums are
unchanged), "Paris" answers are unchanged, chunked prefills stay
bit-identical.

Past the head, the head still costs up to 8% at 4,096 tokens (Qwen2.5-
0.5B 4.74 against 4.38 ms; Mistral 7B 22.02 against 20.82) and about 1%
at 32,768: its chunks score each exact key with one thread reading the
key's f32 row and the unrotated queries from global memory.

A warp per exact key (providers/cuda 292d127, lanes over the dimensions,
sums across the warp) made the grouped code Kernel 47% to 76% slower even
without an exact head (run 44322489467: Qwen2.5-0.5B `tq4` at 32,768 keys
314.3 against 183.6 us, Mistral 7B 1,175.7 against 666.4 us; `tq4` TPOT
10.93 against 7.89 ms), while the unchanged staged Kernel and the f32
Kernel kept their times on the same run. Its registers went from 72 to
78, with no local memory; the cause is not identified. It is reverted:
the exact head's chunks keep a thread per key.
