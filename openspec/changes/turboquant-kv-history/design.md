## Context

In-place KV history (`append-kv-history-in-place`): per layer, one K and one
V store of `[capacity, heads * d]` f32 rows (`KvStoreSpec`). `kv-write`
appends the step's rows at `row_offset`; attention reads the first
`key_count` rows. On CUDA, stores are `CudaDeviceBuffer`s, which already
hold compact forms for weights.

## Decisions

### 1. The quantizer

For each head's row `x ∈ R^d`, with `d` a power of two (64, 128, 256 and
512 in the supported models):

1. `n = ‖x‖₂` (f32). A zero row stores `n = 0`.
2. `y = Π (x / n)`, with `Π = H D / √d`. `H` is the Walsh-Hadamard matrix
   and `D = diag(s)` holds fixed pseudo-random signs (splitmix64 from a
   constant seed and `d`). `Π` is orthogonal and is applied by a fast
   transform in `O(d log d)`.
3. `idx_j` is the nearest centroid of the `b`-bit Lloyd-Max codebook for
   the density of one coordinate of a uniform unit vector in `R^d`,
   `f_d(t) ∝ (1 − t²)^((d−3)/2)` on `[−1, 1]`. The codebook is symmetric
   and is computed by Lloyd iterations on that density in f64 (fixed grid,
   fixed iteration count), so every Provider derives the same `f32` table.
4. The stored value is `n · Πᵀ ĉ`, with `ĉ_j = c[idx_j]`.

The same signs are used for every layer and head: the error guarantee holds
for each vector over the randomness of `Π`, and one fixed `Π` per `d` keeps
the transform a constant.

Alternatives:
- `q4_0` blocks (`quantize-kv-history`): data-independent too, but their
  error grows with the outlier channels of keys, which the rotation spreads
  out.
- Calibrated per-channel K (KIVI): it needs data.

### 2. Precision as a storage choice

`MAGNETAR_KV_PRECISION` = `f32` | `tq4` | `tq3` applies to K and V.
`MAGNETAR_KV_PRECISION_K` / `_V` override each role. The engine resolves it
once per loaded model: a role gets a TurboQuant precision only when:

- the Provider advertises `turboquant-kv-store` on `kv-write`;
- every active attention Kernel advertises `turboquant-kv-read`;
- every layer's head width for that role is a power of two.

Otherwise the role falls back to `f32` and the fallback is recorded.
`KvStoreSpec` carries the resolved precisions, so pooled sets and owned
stores agree. The KV pool keys its sets by precision as well as capacity.

### 3. Store creation and admission

A new `ProviderExecutionApi::write_zeroed_kv_store(id, [capacity, width],
precision, head_dimension)` creates a store in that precision. Its default
returns `Ok(false)`, meaning not supported, and the Runtime then creates an
f32 store as today.

The Memory Manager admits the stored size, `capacity × heads × (d·b/8 + 4)`
bytes. For `d = 128`, `tq4` takes 68 bytes per head and row against 512 in
f32 (7.5× less), and `tq3` takes 52 (9.8× less).

The descriptor the graph sees stays `[capacity, width]` f32.

### 4. Semantics and the Reference CPU

A TurboQuant store holds, at every row, the dequantized value of the last
row written there. The Reference CPU Providers keep f32 storage and write
`dequantize(quantize(row))` in `kv-write` when the store was created in a
TurboQuant precision. Attention is unchanged. Any Provider must compute
attention as on those values, within the f32 tolerance of the conformance
suite.

### 5. CUDA

- **Storage.** `codes` is `u8 [capacity][heads][d·b/8]`, row-major: 4-bit
  codes two per byte, low nibble first, and 3-bit codes eight in three
  bytes. `norms` is `f32 [capacity][heads]`. The codebook (16 or 8 `f32`)
  and the signs live in constant arrays passed to the kernels.
- **`kv-write`.** One block per written row and head:
  - load `d` values into shared memory;
  - reduce for the norm;
  - apply the fast Walsh-Hadamard transform in shared memory, after the
    signs;
  - each thread picks the nearest centroid of its coordinates by
    comparison with the midpoints, then packs and writes.

  The kernel writes at `row_offset + step_offset` (the replay offset), as
  today.
- **Decode attention** (one query row). Per query head:
  - the prologue rotates the query (signs, then the transform);
  - a key's score is `n_k · Σ_j q'_j c[idx_j]`, times the score scale;
  - values accumulate `p · n_v · c[idx_j]` in the rotated domain;
  - the epilogue applies `Πᵀ`, that is the transform and then the signs.

  The kernel follows the split-KV structure of the f32 kernel: chunks of
  keys per block, then a combine kernel. Grouped-query heads share their KV
  head's codes.
- **Prefill and every other read.** The rows `[0, key_count)` are
  dequantized into a scratch f32 buffer of the store's shape, and the
  existing kernel runs. A download, or an expansion for a Kernel without a
  TurboQuant path, dequantizes too. A copy (`copy_tensor_admitted`) copies
  the codes and norms.
- **Step capture.** All launches are capture-safe, with no allocation in a
  captured step. The scratch buffer of a prefill is allocated outside
  capture, like other prefill intermediates.

### 6. Compatibility

The KV layout metadata of a history records its precisions (`quantization:
mode = "tq4"`, …). A history is only continued, reused or restored with the
same precisions, and a mismatch is refused.

## Risks

- **Quality on keys at 3 bits.** If `tq3` keys lose greedy agreement, keep
  `tq4` K with `tq3` V, or add the QJL residual bit for K in a later
  change.
- **Non-power-of-two head widths.** No supported model has one; such a role
  falls back to `f32`.
- **Prefill dequantization cost.** It grows with the context length. It is
  measured, and a prefill kernel reading codes can follow.
