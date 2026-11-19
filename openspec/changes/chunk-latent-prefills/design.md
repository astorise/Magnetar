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

## Evidence

Run 44404969859 (providers/cuda 02b3b21, components/deepseek a79e2ee):
`attention_over_latent_keys_and_values_matches_the_cpu` passes; the tiny
DeepSeek-V2 checkpoints on CUDA match `transformers` (largest logit
difference 9.8e-6). DeepSeek-V2-Lite-Chat Q8_0 across both GPUs, prompts
prefilled in passes of 2,048 rows:

| History | Context | TTFT (ms) | TPOT (ms) | Tokens = f32 | Max \|Δ logit\| step 1 |
|---|---|---|---|---|---|
| f32 | 1,024 | 2,952.3 | 13.39 | - | - |
| f32 | 4,096 | 13,106.3 | 16.41 | - | - |
| f32 | 16,384 | 82,486.6 | 26.91 | - | - |
| f32 | 32,768 | 274,038.5 | 42.98 | - | - |
| `tq4`, exact head 1,024 | 1,024 | 3,641.6 | 15.06 | 64/64 | 0 |
| `tq4`, exact head 1,024 | 4,096 | 18,199.1 | 19.84 | 64/64 | 7.10 |
| `tq4`, exact head 1,024 | 16,384 | 113,112.7 | 34.60 | 64/64 | 9.17 |
| `tq4`, exact head 1,024 | 32,768 | 327,023.8 | 52.76 | 64/64 | 13.78 |
| `tq4`, no exact head | 1,024 | 4,160.1 | 15.59 | 64/64 | 8.39 |
| `tq4`, no exact head | 4,096 | 19,357.2 | 20.39 | 64/64 | 11.18 |
| `tq4`, no exact head | 16,384 | 118,196.1 | 34.96 | 64/64 | 12.49 |
| `tq4`, no exact head | 32,768 | 329,956.8 | 52.01 | 64/64 | 14.98 |

An `f32` history now reaches 16,384 and 32,768 tokens (run 44389385779: out
of memory at 16,384 in one pass). With a `tq4` history the first step's
logits differ from `f32`'s (decision 4: the prefill reads the history's
round trip; with every prompt row in the exact head they are equal), the
64 tokens still equal. The `tq4` prefill takes longer than the `f32` one
(3,641.6 against 2,952.3 ms at 1,024 tokens in one pass, before any
chunking); `latent_prefill_attention_times` times the pass over each
history. This run's decode on codes is the slower four-way split of
providers/cuda 27e1b9e (`cache-latent-attention-history`).

`latent_prefill_attention_times` (run 44417598937, providers/cuda 6e267f2):
one latent prefill attention pass of 2,048 queries takes the same time over
an `f32` history and a `tq4` one (23.34 against 22.45 ms after 2,048 keys,
153.37 against 154.13 after 8,192, 335.52 against 336.44 after 16,384), its
split and dequantization at most 1.05 ms of it: the `tq4` prefill's extra
time (3,975.4 against 2,960.4 ms at 1,024 tokens in that run) is spent
elsewhere. The pass itself runs the tiled 192/128 Kernel at about 0.5
multiply-add Tera-operations per second (2,048 x 16,384 x 16 heads x 320 in
335 ms): at 32,768 tokens it is about half of TTFT.

The DeepSeek phases step (run 44431584039: every dispatch waits for its
Kernels, decode steps node by node, `f32` then `tq4` without an exact head)
attributes each request's host time, which then covers Device time:

| Phase | f32, 1,024 | tq4, 1,024 | f32, 16,384 | tq4, 16,384 |
|---|---|---|---|---|
| prefill | 3,292.6 ms | 3,651.7 ms | 94,919.4 ms | 106,225.4 ms |
| `moe-experts` | 2,124.4 | 2,359.3 | 35,213.7 | 40,901.0 |
| `gated-matmul` | 338.4 | 382.3 | 5,217.0 | 6,394.8 |
| `matmul-residual` | 292.8 | 324.1 | 4,203.5 | 4,731.6 |
| `matmul` | 232.3 | 254.3 | 3,165.7 | 3,704.5 |
| `attention` | 229.7 | 256.0 | 46,196.1 | 49,505.1 |
| KV append | 2.2 | 7.5 | 22.3 | 59.1 |
| decode step | 35.66 ms | 36.69 ms | 52.02 ms | 49.45 ms |
| its `attention` | 1.59 | 2.24 | 15.46 | 12.88 |

The `tq4` prefill's extra time is spread over every operator, those that
never read the history included (`moe-experts` 11% and 16% slower), while
its own work, the KV append, adds 5 and 37 ms. That is not TurboQuant work;
it is consistent with the Device running slower later in the job (the
`tq4` requests follow the `f32` ones, 32,768 tokens included, in every run)
while the bandwidth-bound decode steps keep their times; this run does not
measure the clocks. Run 44441670479 does: the first GPU's SM clock stays
between 1,732 and 1,942 MHz and its temperature at most 64 C through the
`f32` and `tq4` requests alike, while the `tq4` TTFTs are again 22% (1,024
tokens) to 19% (32,768) above `f32`'s, even with every 1,024-token prompt
row in the exact head (the same values as `f32`'s): the clocks do not
explain it. A `tq4` decode step at 16,384 tokens takes 2.57 ms less
than an `f32` one, its attention's 2.58 ms. At 16,384 tokens the latent
prefill attention is 49% of the prefill and `moe-experts` 37%.

Run 44450645042 runs `f32`, `tq4` (no exact head) and `f32` again, each a
model of its own, every dispatch waiting for its Kernels: prefill 3,295.4,
3,399.4 and 3,560.6 ms at 1,024 tokens, 14,937.7, 15,474.1 and 16,904.8 ms
at 4,096, the decode steps 36 to 40 ms in all three. Every model loaded
after the first prefills slower, whatever its history: the `tq4` TTFTs
measured after `f32`'s carry it. providers/cuda ed7edfd keeps the memory
the stream-ordered allocator frees in the device's pool instead of
returning it to the driver at each synchronization, which a prefill's
large temporaries met at every pass.
