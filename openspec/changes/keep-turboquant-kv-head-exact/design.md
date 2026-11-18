## Context

`turboquant-kv-history` stores each row's heads as TurboQuant codes and an
f32 norm, against per-column offsets set by the write at row 0. Decode
attention runs `tq_attention_split_*_kernel` on the codes (scores against
the rotated query, values accumulated in rotated space minus the value
offsets), `attention_combine_kernel` and `tq_rotate_back_kernel` (which
adds the value offsets back). Prefill reads go through a scratch buffer
the store is dequantized into. The CPU Providers hold a store as f32 rows
that `TurboQuantStoreState::write` replaces by their round trips.

## Decisions

### 1. The head, not a window

Rows `[0, E)` of a store are exact; rows from `E` on are quantized. The
head never moves: no ring buffer, no eviction, and no interaction with the
padding rows a bucketed or chunked prefill writes beyond the prompt and
decode later writes over (those lie at the end of the history). It holds
the sinks of every history, and the whole of a history of up to `E` rows,
which therefore decodes as f32 does.

`E` defaults to 1,024 rows: Mistral 7B's 64 stores hold 1,024 rows of
1,024 f32 values, 256 MiB; Qwen2.5-0.5B's 48 stores of 128 values, 24 MiB.

### 2. Semantics

A store row read by attention at history length `L` is the written row
itself when its index is below `E`, else its TurboQuant round trip with
the store's offsets. `TurboQuantStoreState::write` returns rows below `E`
unchanged. The offsets are still set by the write at row 0, which may lie
entirely in the head: they apply only to rows from `E` on.

### 3. CUDA storage and writes

`CudaTqStore` gains `exact: [E][width]` f32 (`E` at most the capacity).
`tq_kv_write_kernel` copies a row below `E` there as it is, at the
captured row offset like any row, and quantizes the others (the codes of
rows below `E` are never written or read): one Kernel per write, as
before.

### 4. CUDA decode by regime

- **`L <= E`**: the f32 decode attention on the heads, `key_count = L`,
  exactly as for f32 stores. Inside a step capture it declares `E - L` as
  its largest replay offset (the heads' rows less the key count): the
  first step past `E` leaves the captured bounds, runs per node, and a
  later step is captured in the other regime. A history crosses `E` once.
- **`L > E`**: the code split Kernels read a chunk's keys below `E` (a
  prefix of the chunk, which a sliding window may start inside the head)
  from the heads: such a key is scored against the query as it is
  (`q . k`), and its normalized weight times its value is summed
  unrotated, apart from the codes' rotated sum. Once per block that
  unrotated sum, less the value offsets times its weights' sum, is
  rotated (`Pi (sum_j w_j (v_j - o_v))`) and added to the block's
  partial, which is then in the codes' domain: the combine and the
  inverse rotation (which adds the offsets back) are unchanged. This
  replaces a separate head pass and partial move: no extra Kernel, one
  Walsh-Hadamard transform per block that reads the head, and no
  constraint tying `E` to the chunk length.

### 5. Prefill reads

A prefill whose rows the heads hold attends on them directly, as on f32
stores; a longer one reads the dequantized scratch, which takes rows
below `E` from the heads.

### 6. Configuration

`MAGNETAR_KV_EXACT_ROWS` (1,024; `0` off) and `KvPrecisions::exact_rows`
(`KvPrecisions::uniform` takes the default head, `with_exact_rows` another,
for `ProductionLoadedModel::request_kv_precisions`), resolved with the
precisions and recorded with them in the KV compatibility
("turboquant-kv:k=tq4,v=tq4,exact=1024") and layout metadata. f32 stores
have no head. A Provider keeps a head when it advertises
`turboquant-kv-exact-head` on `kv-write`; otherwise the head falls back to
none, with the reason recorded, and every row is quantized.

## Risks

- Decode past `E` reads `E` f32 rows per KV head where the codes-only
  decode reads codes, and rotates one sum per block that reads them:
  measured against the codes-only decode at 4,096 to 32,768 tokens.
- Memory: `E` f32 rows per store, reported with the KV bytes per token.
