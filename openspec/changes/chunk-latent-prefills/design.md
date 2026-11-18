## Context

`chunk-long-prefills` runs a long prompt as passes of one graph of the chunk
length; a pass continues the KV history in place, its attention reading the
history's first `key_count` rows. It applies to a graph whose every
`attention` reads its K and V from KV-history edges
(`prefill_continues_history`). The DeepSeek Component's prefill attended
with keys `[c_kv W_UK | k_pe]` and values `c_kv W_UV` computed from its own
rows by two matmuls, beside the latent history it wrote: it never chunked.

## Decisions

### 1. An attention form, not a new Component export

A Component export for "continuation" graphs would change the
`model-component-graph` world (every Component rebuilt), and its later
passes would still need keys and values for the history's rows. The
graph instead names the history as K and V and gives the attention the two
projections; whoever executes the node expands the rows it reads. Only the
operator catalogue (`attention` takes 3 to 5 inputs), the Providers and the
DeepSeek Component change; `magnetar:model-component-graph@1.7.0` stays.

### 2. Semantics

`attention` with `latent_kv_width = L` and inputs `(q, k, v, w_k, w_v)`: the
first `key_count` rows of `k` and `v` (every row without it) are latent rows
of `L` values `c` and a rope key the heads share; head `h`'s key is
`[c W_k_h | k_rope]` (`w_k [L, head_count * (head_dimension - rope)]`) and
its value `c W_v_h` (`w_v [L, head_count * value_head_dimension]`). The
attention is then the expanded one: `kv_head_count = head_count`, the
queries the last rows, causal. `operator_semantics::latent_keys` and
`latent_values` give the expansion; each product sums in index order, as
`matmul` does, so on the Reference CPU an f32 history gives the values the
two matmul nodes gave.

### 3. Expanding every row a pass reads

A pass expands all `key_count` rows, the earlier passes' included: per pass
`key_count x L x head_count x (nope + value)` multiply-adds, about a fifth of
the pass's attention over those rows at DeepSeek-V2-Lite's shapes (1,088
against 320 per query, key and head for the absorbed form instead). On CUDA
the rows go through the Provider's matmul (packed weights read packed, long
products in row blocks) and head concatenation, then the tiled 192/128
attention; a TurboQuant history is dequantized for the call. The expanded
keys and values of the rows read are held during the node only.

### 4. TurboQuant histories in the prefill

With the history the attention's K and V, a prefill reads the history's
stored values: a TurboQuant history's round trip, as a chunked prefill of a
standard model reads its TurboQuant store, and as every decode step already
reads it. The previous latent prefill attended over its rows' exact keys.

### 5. Decode unchanged

A decode step keeps the absorbed form: one query row against the latent
history in the latent width.

## Risks

- **A step capture.** The expansion allocates, so a captured prefill
  declines (`forbid_in_capture`), as the expert prefill's scratch already
  does at more than 64 tokens.
- **Expansion memory.** A pass holds the expanded keys and values of every
  row it reads: 20 KB per row at DeepSeek-V2-Lite's shapes (335 MB at 16,384
  rows), within a layer's node.
