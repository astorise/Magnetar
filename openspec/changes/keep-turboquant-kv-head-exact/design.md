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

`CudaTqStore` gains `exact: [E][width]` f32. `kv-write` copies the rows
below `E` there (`tq_exact_write_kernel`, which reads the captured row
offset like `tq_kv_write_kernel`) and quantizes the others (the codes of
rows below `E` are never read).

### 4. CUDA decode by regime

- **`L <= E`**: the f32 split-KV attention on the head, `key_count = L`.
  Inside a step capture it declares `E - L` as its largest replay offset:
  the first step past `E` leaves the captured bounds, runs per node, and a
  later step is captured in the other regime. A history crosses `E` once.
- **`L > E`**: the f32 split kernel scores the head's chunks, then
  `tq_head_partials_kernel` moves its partials into the codes' space
  (subtracts the value offsets weighted by the chunk's sum and rotates);
  the code kernel scores chunks from `E / ATTENTION_CHUNK_KEYS` on into the
  same partials buffer; one combine and one inverse rotation follow. `E`
  is a multiple of the chunk length.

### 5. Prefill reads

The dequantized scratch takes rows below `E` from the head.

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

- Decode past `E` runs one more Kernel pair than today (the head's split
  and partial move): measured against the codes-only decode at 4,096 to
  32,768 tokens.
- Memory: `E` f32 rows per store, reported with the KV bytes per token.
