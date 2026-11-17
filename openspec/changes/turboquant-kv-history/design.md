## Context

In-place KV history (`append-kv-history-in-place`): per layer, one K and one
V store of `[capacity, heads * d]` f32 rows (`KvStoreSpec`). `kv-write`
appends the step's rows at `row_offset`; attention reads the first
`key_count` rows. On CUDA, stores are `CudaDeviceBuffer`s, which already
hold compact forms for weights.

## Decisions

### 1. The quantizer

Each store keeps one f32 offset per column of its rows, `μ`: the mean of
the rows of the write that starts its history (the write at row 0, summed
in row order in f32), zeros before it. For each head's row `x ∈ R^d`, with
`d` a power of two (64, 128, 256 and 512 in the supported models), the
store quantizes `x − μ` (that head's offsets):

1. `n = ‖x − μ‖₂` (f32). A zero difference stores `n = 0`.
2. `y = Π (x / n)`, with `Π = H D / √d`. `H` is the Walsh-Hadamard matrix
   and `D = diag(s)` holds fixed pseudo-random signs (splitmix64 from a
   constant seed and `d`). `Π` is orthogonal and is applied by a fast
   transform in `O(d log d)`.
3. `idx_j` is the nearest centroid of the `b`-bit Lloyd-Max codebook for
   the density of one coordinate of a uniform unit vector in `R^d`,
   `f_d(t) ∝ (1 − t²)^((d−3)/2)` on `[−1, 1]`. The codebook is symmetric
   and is computed by Lloyd iterations on that density in f64 (fixed grid,
   fixed iteration count), so every Provider derives the same `f32` table.
4. The stored value is `μ + n · Πᵀ ĉ`, with `ĉ_j = c[idx_j]`.

The same signs are used for every layer and head: the error guarantee holds
for each vector over the randomness of `Π`, and one fixed `Π` per `d` keeps
the transform a constant.

Why the offsets: the quantization error of a row is proportional to the
norm it quantizes. Keys whose rows share a large component (Qwen2.5's key
projection has a bias, and keys have outlier channels) spend the codes on
that component and lose the part that differs from key to key. Qwen2.5-0.5B
Q8_0 on the CPU (prompt "The capital of France is", 64 greedy tokens)
continues " Paris." in f32, in `tq4` V with f32 K, but not in `tq4` K
without offsets (" the, the, the, …"; `tq4` K and V gives
"\n\n10.\n\n10.0.10…"). With the offsets, `tq4` and `tq3` K and V
continue " Paris.". The GPU runner found the same before the offsets (run
44128259285, CUDA, 64 greedy tokens against f32): Qwen2.5-0.5B lost
" Paris." in `tq4` (0/64 leading tokens agree) and `tq3` (0/64), Gemma 4
E2B kept it (4/64 each: f32 itself ends after " Paris.\n\n"), and
Mistral 7B Q4_K_M agreed on 48/64 tokens in `tq4` and opened with
"a city that is full of history" in `tq3`. With the offsets (run
44138881338), all three continue with "Paris" in `tq4` and `tq3`; the
leading tokens that agree with f32 are Qwen2.5-0.5B 8/64 (`tq4`) and
2/64 (`tq3`), Gemma 4 E2B 7/64 and 12/64, Mistral 7B 13/64 and 1/64.
Greedy agreement counts end at the first near-tie, so they vary from run
to run (Mistral `tq4` agreed on 48 tokens without offsets); the
first-step logit differences of the gains test measure the error more
steadily. Attention is exact under any fixed offsets: a key's
offset adds `q · μ` to every score of a query, which the softmax ignores,
and values read back with their offset added. Only the quantization error
changes.

Alternatives:
- `q4_0` blocks (`quantize-kv-history`): data-independent too, but their
  error grows with the outlier channels of keys, which the rotation spreads
  out.
- Calibrated per-channel K (KIVI): it needs calibration data.
- Outlier channels at more bits (the paper's 2.5/3.5-bit mixes): it needs
  the outlier channels chosen per model; the offsets need no choice.

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
row written there. The Reference CPU Providers keep f32 storage and the
store's offsets (`TurboQuantStoreState`); `kv-write` into a store created in
a TurboQuant precision first sets the offsets when it writes at row 0, then
writes `μ + dequantize(quantize(row − μ))`. Attention is unchanged. Any Provider must compute
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
  today. A write at row 0 first sets the offsets (`tq_offsets_kernel`, one
  thread per column summing the rows in order). A replayed bucketed
  prefill does not compute its trailing padding rows, so the offsets
  average only the rows it computes (`padding_rows`); a prefill run per
  node computes and averages them all. `offsets` is `f32 [heads][d]`.
- **Decode attention** (one query row). Per query head:
  - the prologue rotates the query (signs, then the transform) and
    computes `q · μ_k` for its KV head;
  - a key's score is `q · μ_k + n_k · Σ_j q'_j c[idx_j]`, times the score
    scale;
  - values accumulate `p · n_v · c[idx_j]` in the rotated domain;
  - the epilogue applies `Πᵀ`, that is the transform and then the signs,
    and adds `μ_v` (the weights sum to one).

  The kernel follows the split-KV structure of the f32 kernel: chunks of
  keys per block, then a combine kernel. Grouped-query heads share their KV
  head's codes.
- **Prefill and every other read.** The rows `[0, key_count)` are
  dequantized into a scratch f32 buffer of the store's shape, and the
  existing kernel runs. A download, or an expansion for a Kernel without a
  TurboQuant path, dequantizes too. A copy (`copy_tensor_admitted`) copies
  the codes, norms and offsets.
- **Step capture.** All launches are capture-safe, with no allocation in a
  captured step. The scratch buffer of a prefill is allocated outside
  capture, like other prefill intermediates.

### 6. Compatibility

A prefill records its stores' precisions in the history's KV
compatibility (`quantization_mode`) and layout metadata (`quantization`),
as `turboquant-kv:k=<precision>,v=<precision>`; f32 stores record none,
as before. A later step validates the history against the precisions this
model's stores resolve to then, and refuses a mismatch as incompatible
(`kv cache dtype mismatch`). The KV pool keeps each set's precisions and
reuses a set only for requests in the same precisions.

## Risks

- **Quality on keys at 3 bits.** If `tq3` keys lose greedy agreement, keep
  `tq4` K with `tq3` V, or add the QJL residual bit for K in a later
  change.
- **Offsets from a short or unrepresentative prefill.** The offsets come
  from the prefill's rows alone. They stay exact for attention whatever
  they are; a history whose later rows drift away from them quantizes
  larger differences.
- **Non-power-of-two head widths.** No supported model has one; such a role
  falls back to `f32`.
- **Prefill dequantization cost.** It grows with the context length. It is
  measured, and a prefill kernel reading codes can follow.
