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
2. `q̃ = batched-matmul(q_nope [1, H·n], k_absorb [H·n, L])`, `[1, H·L]`;
3. `q = concat_heads(q̃, q_pe)`, `[1, H·(L + r)]`;
4. the latent row `[c_kv | k_pe]` (`concat_heads` of one head each),
   marked as the K history;
5. `attention(q, latent, latent)` with `head_count = H`,
   `kv_head_count = 1`, `head_dimension = L + r`,
   `value_head_dimension = L`, `value_head_stride = L + r`, and the
   expanded form's score scale (`softmax_scale(n + r)`, YaRN included):
   the scores are the expanded form's;
6. `o = batched-matmul(attention [1, H·L], v_absorb [H·L, v])`,
   `[1, H·v]`, then `o_proj`.

`k_absorb`'s group `h` is `W_UK_hᵀ` (`[n, L]`); `v_absorb`'s is `W_UV_h`
(`[L, v]`).

### 3. Expanded prefill

The prefill graph keeps today's attention on expanded keys and values of
its own rows, and adds the latent row node marked as the K history, which
nothing else consumes. Prefill attention then costs `H·(n + r + v)` per
key pair rather than `H·(2L + r)`; a prefill always starts its history at
row 0 here, so it never needs the latent of earlier rows.

### 4. `batched-matmul`

`a [T, G·K]`, `b [G·K, N]`, attribute `group_count = G`: for each group
`g`, `out[:, g·N..(g+1)·N] = a[:, g·K..(g+1)·K] · b[g·K..(g+1)·K, :]`.
f32 weights; the two absorbed tables are small (`H·n·L` and `H·L·v`
values per layer, 1 Mi each in V2-Lite).

### 5. `attention.value_head_stride`

V rows hold `kv_head_count` heads of `value_head_stride` values; a KV
head's value is the first `value_head_dimension` of its head.
Default: `value_head_dimension`. With V being the K history, stride
`L + r` and dimension `L` read `c_kv`.

### 6. Absorbed weights

The loaders already split the latent up-projection into `k_nope_proj` and
`v_proj` (`kv_b_proj`, or llama.cpp's `attn_k_b` / `attn_v_b`). They also
emit `k_absorb` and `v_absorb`, as f32: a block transpose of `k_nope_proj`
is not a byte copy of a quantized table, so packed tables are dequantized
first (exact values of the stored weights).

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
