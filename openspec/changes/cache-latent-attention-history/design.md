## Context

The DeepSeek Component expands MLA (`express-latent-attention-forms`). Per
layer, with `H` heads, `n = qk_nope_head_dim`, `r = qk_rope_head_dim`,
`v = v_head_dim` and `L = kv_lora_rank`:

- `c_kv = rmsnorm(h · kv_a_proj)` (`[T, L]`), `k_pe = rope(h · k_rope_proj)`
  (`[T, r]`, one head);
- `k = [c_kv · k_nope_proj | k_pe]` per head (`[T, H·(n + r)]`), the K
  history; `v = c_kv · v_proj` (`[T, H·v]`), the V history;
- `attention(q, k, v)` with `value_head_dimension = v`.

Weights are `[in, out]`: `k_nope_proj` is `[L, H·n]`, `v_proj` `[L, H·v]`.
The engine writes the output of the node marked K (V) into the layer's K
(V) store and gives its consumers the store (`append-kv-history-in-place`).

## Decisions

### 1. The latent history

A latent layer's K history is `[c_kv | k_pe]`, `L + r` values per token
(576 in V2-Lite): one key head. It has no V history: decode attention
reads the values from the key rows. The engine creates no V store for such
a layer and its KV state names the K store for both roles.

`FirstNativeModelConfig` gives a latent layer K rows of `L + r` and no V
rows; `kv_bytes_per_token`, the pool and the layout metadata follow.

### 2. Absorbed decode

For query head `h`, with `W_UK_h = k_nope_proj[:, h·n..(h+1)·n]` (`[L, n]`)
and `W_UV_h = v_proj[:, h·v..(h+1)·v]` (`[L, v]`):

- `q_nope_h · (c_kv W_UK_h)ᵀ = (q_nope_h W_UK_hᵀ) · c_kv`: the absorbed
  query `q̃_h = q_nope_h W_UK_hᵀ` (`[L]`) scores against the latent;
- `o_h = Σ_t p_t (c_kv,t W_UV_h) = (Σ_t p_t c_kv,t) W_UV_h`.

The decode graph:

1. `q_nope`, `q_pe` as today (`q_pe` rotated);
2. `q̃ = batched-matmul(q_nope [1, H·n], k_absorb [H·n, L])`, `[1, H·L]`
   (group `h` of `k_absorb` is `W_UK_hᵀ`, `[n, L]`);
3. `q = concat_heads(q̃, q_pe)`, `[1, H·(L + r)]`;
4. the latent row `[c_kv | k_pe]` (`concat_heads` of one head each),
   marked as the K history;
5. `attention(q, latent, latent)` with `head_count = H`,
   `kv_head_count = 1`, `head_dimension = L + r`,
   `value_head_dimension = L`, `value_head_stride = L + r`, and the
   expanded form's score scale (`softmax_scale(n + r)`, YaRN included):
   the scores are the expanded form's;
6. `o = batched-matmul(attention [1, H·L], v_absorb [H·v, L],
   transpose_b)`, `[1, H·v]` (group `h` of `v_absorb` is `W_UV_hᵀ`,
   `[v, L]`, read transposed), then `o_proj`.

Both tables are the latent up-projection's rows as checkpoints store them
(`[out, in]`): the key rows and the value rows of each head.

### 3. Expanded prefill

The prefill graph keeps today's attention on expanded keys and values of
its own rows, and adds the latent row node marked as the K history, which
nothing else consumes. Prefill attention then costs `H·(n + r + v)` per
key pair rather than `H·(2L + r)`; a prefill always starts its history at
row 0 here, so it never needs the latent of earlier rows.

### 4. `batched-matmul`

`a [T, G·K]`, attribute `group_count = G`: for each group `g`,
`out[:, g·N..(g+1)·N] = a[:, g·K..(g+1)·K] · B_g`, with `B_g =
b[g·K..(g+1)·K, :]` (`b [G·K, N]`) or, with `transpose_b`,
`b[g·N..(g+1)·N, :]ᵀ` (`b [G·N, K]`). f32 weights; the two absorbed tables
are small (`H·n·L` and `H·v·L` values per layer, 1 Mi each in V2-Lite).

### 5. `attention.value_head_stride`

V rows hold `kv_head_count` heads of `value_head_stride` values; a KV
head's value is the first `value_head_dimension` of its head.
Default: `value_head_dimension`. With V being the K history, stride
`L + r` and dimension `L` read `c_kv`.

### 6. Absorbed weights

The loaders already split the latent up-projection into `k_nope_proj` and
`v_proj` (`kv_b_proj`, or llama.cpp's `attn_k_b` / `attn_v_b`) by rows.
They also emit the same key rows as `k_absorb` and value rows as
`v_absorb`, under names that are not projections, so they keep the stored
`[out, in]` orientation: row selections, exact byte copies even of
quantized tables. A Provider whose `batched-matmul` reads f32 gets them
dequantized.

### 7. Compatibility

The geometry changes the history's layout; a history written by the
expanded form is not continued by the latent one. The Component's version
and digest change, and the KV compatibility key carries both.

## Risks

- **Decode cost per key.** Absorbed attention reads 576 values per key and
  computes `H·(2L + r)` products, against 5,120 values and `H·(n + r + v)`
  expanded. Decode is bound by the bytes it reads, but short histories may
  be slower; measured on V2-Lite.
- **Numerics.** The absorbed scores sum in another order than the expanded
  ones; the tiny checkpoints' tolerances against `transformers` hold or the
  difference is reported.

## Evidence: latent decode on codes

One RTX 3060; one decode step's absorbed attention at DeepSeek-V2-Lite's
shapes (16 query heads over one history of 576-value rows, the values
each row's leading 512, `tq4` in groups of 64), `latent_decode_attention_times`:

| us | 1,024 keys | 4,096 | 16,384 | 32,768 |
|---|---|---|---|---|
| f32 history | 101.6 | 318.7 | 3,030.1 | 6,886.6 |
| `tq4` through dequantized scratch (before) | 149.7 | 513.1 | 3,688.3 | 8,105.5 |
| `tq4` on its codes | 105.6 | 154.8 | 525.4 | 921.5 |
| `tq4` on its codes, 1,024-row exact head | 108.6 | 300.9 | 616.7 | 904.4 |

(Run 44352387517, providers/cuda 472bf54; run 44351792640 measured the
first two rows the same.) The decode steps read each row's leading values
in place (providers/cuda c8abbd1) instead of gathering the history at
every step. On its codes (`tq_latent_attention_split_kernel`: a block per
chunk and eight query heads, each code decoded once for them, the values
summed in each group's rotated domain with the group's norm) a `tq4`
latent decode is 8.8x faster than through dequantized scratch at 32,768
keys and 7.5x faster than the f32 history's, whose per-head Kernel reads
the history once for each of the 16 heads (11 GB/s). Conformance:
`turboquant_latent_decode_on_codes_matches_the_cpu` (`tq4` and `tq3`, 33
to 1,100 keys, exact heads, a sliding window) and
`batched_matmul_and_strided_values_match_the_cpu` pass.

Run 44355981594 (providers/cuda 8947b7f): the f32 latent decode takes a
block per chunk and eight query heads (`latent_attention_split_kernel`),
reading each row once for them all, the values in place:

| us | 1,024 keys | 4,096 | 16,384 | 32,768 |
|---|---|---|---|---|
| f32 history, per-head Kernel (before) | 101.6 | 318.7 | 3,030.1 | 6,886.6 |
| f32 history, eight heads per block | 101.2 | 163.6 | 810.4 | 1,499.6 |
| `tq4` on its codes | 107.9 | 155.2 | 560.6 | 1,046.2 |
| `tq4` on its codes, 1,024-row exact head | 105.3 | 295.0 | 605.0 | 1,122.8 |

The f32 latent decode is 4.6x faster at 32,768 keys (50 GB/s against
11); `tq4` on its codes stays 1.4x faster than it there. At 1,024 keys
both take about 100 us: 8 chunks and 2 head blocks give 16 blocks for 28
multiprocessors. `latent_decode_matches_the_cpu` (33 to 1,100 keys, a
sliding window) passes.
