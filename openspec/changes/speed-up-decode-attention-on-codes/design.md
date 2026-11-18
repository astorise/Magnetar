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
when the last wave would be less than three quarters full.

## Risks

- Shared memory for staged codes: 128 keys x (2 x 64 + 8) bytes, about
  17 KB for 128-wide `tq4` heads; with decision 3 the total stays below
  today's.
- Summation order changes: results move within f32 rounding; the
  conformance tolerances are unchanged.
