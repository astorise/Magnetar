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

## Evidence

One RTX 3060 (12 GB). Run 44289082020 (providers/cuda c0b093f): greedy
decoding of 64 tokens after prompts of 512 to 32,768 tokens, `tq4` with
the default head of 1,024 rows and with none, `tq3` with the head; run
44283381146 (the commit before) measured `tq4` and `tq3` without a head.

| TPOT vs f32 | 512 | 4,096 | 16,384 | 32,768 |
|---|---|---|---|---|
| Qwen2.5-0.5B, `tq4` with the head | 1.01x | 0.91x | 0.96x | 1.01x |
| Qwen2.5-0.5B, `tq4` without | 0.81x | 0.92x | 0.96x | 1.01x |
| Qwen2.5-0.5B, `tq3` with the head | 1.01x | 0.87x | 0.85x | 0.83x |
| Mistral 7B Q4_K_M, `tq4` with the head | 0.99x | 0.99x | 1.12x | - |
| Mistral 7B Q4_K_M, `tq4` without | 0.95x | 1.02x | 1.14x | - |
| Mistral 7B Q4_K_M, `tq3` with the head | 0.99x | 0.94x | 0.98x | - |

(Mistral 7B's f32 history does not fit at 32,768 tokens; `tq4` decodes
there at 53.8 ms with the head and 53.1 ms without.)

- **Up to the head, f32's speed and f32's logits.** At 512 tokens every
  history lies in the heads: TPOT is f32's (3.65 against 3.69 ms; 17.98
  against 17.86 ms) and the first step's logits are f32's exactly (largest
  difference 0, against 3.2 and 1.1 without the head).
- **Past the head, `tq4` within 3% of the codes-only decode** (0.3% to
  1.5% on Qwen2.5-0.5B, 1.3% to 3% on Mistral 7B).
- **`tq3` with the head was 8% to 24% slower past it than `tq3` without in
  run 44283381146** (Qwen2.5-0.5B at 32,768 tokens: 11.25 against
  9.04 ms), a gap that grows with the history where `tq4`'s does not. A
  suspected cause: the grouped code Kernel stored the exact head's sums
  through a run-time head index, which can move its accumulators from
  registers to local memory in every loop. providers/cuda 5164290 unrolls
  that loop with constant indices; to be measured, with each Kernel's
  local memory reported.
- **Agreement.** All 64 tokens agree with f32 for every precision and
  context but `tq3` at 16,384 tokens on Qwen2.5-0.5B (32/64). On Mistral 7B
  the head brings the first step's largest logit difference from 2.24 to
  0.42 at 4,096 tokens and from 1.88 to 0.98 at 16,384; on Qwen2.5-0.5B it
  stays within the codes-only run's (10.5 against 9.6, 7.6 against 7.8,
  7.0 against 7.2).

Kernel times (run 44290046381, `decode_attention_times`, one decode step's
attention): with the head, 512 keys take f32's time (Qwen2.5-0.5B 12.7 us,
against 34.6 us on codes; Mistral 7B 35.7 against 45.8 us).
