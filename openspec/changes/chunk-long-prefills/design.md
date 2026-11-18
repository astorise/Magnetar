## Context

A first-native prefill runs the Component's prefill graph built for the
prompt's length: every node processes every prompt row at once. With KV
history kept in place, the prefill creates one store per layer and role of
`prompt + max_new_tokens` rows and writes its rows from row 0; decode steps
continue the stores at `row_offset` = the committed length, with the RoPE
position overridden and attention bounded by `key_count`.

## Decisions

### 1. A pass is a prefill that continues a history

The engine already has every piece a later pass needs:

- `execute_first_native_graph_*` with a `kv_history` whose layers have
  `rows: Some(p)` continues those stores: its K and V rows are written at
  `row_offset = p`, and the attention reads `key_count = p + chunk` rows.
- The absolute position override sets every RoPE node's `position_offset`
  to `p`: the pass's rows take positions `p..p + chunk`.
- Attention places its query rows at the end of the keys (causal offset
  `key_count - rows`), on the Reference CPU and in the CUDA tiled and
  Tensor Core Kernels: row `i` of the pass sees keys `0..=p + i`.

So pass `k` runs the chunk's prefill graph with the history the passes
before it left and position `p = k * chunk`. The first pass starts the
stores (no history), as a one-pass prefill does.

### 2. One graph: the last pass is padded

The chunk's graph is prepared like a bucket's (`replay-bucketed-prefill`):
with the logits row gather when its LM head takes it. The last pass of `r`
real rows is right-padded to the chunk length with its last token and
selects row `r - 1`. Padding rows come after every real row, so no real row
reads them; they are written to the stores beyond the prompt, and the
history's row count is set to the prompt's length, so decode writes over
them. The stores are created with `max(prompt + max_new_tokens,
ceil(prompt / chunk) * chunk)` rows.

All passes, and every prompt longer than the chunk, share one cached graph
and one Prepared Execution Plan. Every pass selects one logits row (the LM
head runs on one row); only the last pass's are the prefill's.

The generation loop checks a prefill's plan against the whole prompt
(`PlanGuard::SequenceRange`), so the plan of the chunk's graph declares
that it serves prompts of any length, while every other plan still serves
up to its own rows. The first GPU run found this (run 44251614646: every
chunked prompt longer than 32 tokens, the tiny fixtures' floor, was
refused with `PlanWorkloadIncompatible`).

### 3. Eligibility is structural

`ComponentGraphCache::prefill_graph_length` returns the chunk length for a
prompt longer than it (never under tensor parallelism). Before a request,
the production path checks the chunk graph:

- every `attention` node reads its K and V inputs from KV-history edges;
- those edges feed nothing but attention (the bucket condition).

A graph failing either runs every prompt in one pass from then on (the chunk
length is set to 0). An expanded latent attention reads K and V computed
from its own rows while its history holds the latent rows, so it fails the
first condition: a later pass could not read the earlier passes' keys. A
Provider without in-place history never chunks.

### 4. Prompts of exactly the chunk's length

They run the chunk's graph in one pass. They are not a bucket's: their
stores are their own, not pooled, and their prefill is not recorded.

### 5. Default

2,048 rows. A pass then holds, for Mistral 7B, a 117 MB MLP intermediate
instead of 940 MB at 16,384 tokens. Prompts up to 2,048 tokens are
unchanged.

## Risks

- **Re-reading the history.** Each pass's attention reads the whole history
  so far, as the one-pass prefill's rows do in total; the CUDA Tensor Core
  attention also splits the history's K and V into bf16 terms per pass
  (`O(prompt² / chunk)` conversions), and a TurboQuant history is
  dequantized per pass. Measured as TTFT against one pass.
- **Numerics.** Passes of another row count may take other matmul Kernels;
  they never change positions or keys. The CUDA test bounds the first-step
  logits difference.
