## Why

Decode attention over TurboQuant stores (`turboquant-kv-history`) reads
7 to 10 times fewer bytes than over f32 stores, but it is not faster in
proportion. One decode step's attention on the RTX 3060
(`decode_attention_times`, run 44290046381):

| Keys | Qwen2.5-0.5B f32 | Qwen2.5-0.5B `tq4` | Mistral 7B f32 | Mistral 7B `tq4` |
|---|---|---|---|---|
| 4,096 | 44.4 us (95 GB/s) | 43.6 us (14 GB/s) | 185.9 us (181 GB/s) | 165.7 us (27 GB/s) |
| 16,384 | 136.3 us (123 GB/s) | 145.9 us (16 GB/s) | 689.7 us (195 GB/s) | 545.9 us (33 GB/s) |
| 32,768 | 253.3 us (132 GB/s) | 249.0 us (19 GB/s) | 1,387.2 us (194 GB/s) | 1,060.5 us (34 GB/s) |

The f32 Kernels reach 130 to 195 GB/s of the bytes they must read; the
code Kernels reach 14 to 34 GB/s. At 32,768 tokens a Qwen2.5-0.5B decode
step's 24 attentions take about 6 ms of its 9.3 ms (run 44289082020), in
either precision.
The codes' Kernel is bound by how it reads and works, not by the bytes.

The Gemini RFC's section 4 proposes a split-K schedule for this Kernel:
a bounded number of blocks per KV head, each walking many chunks, set
from the Device's multiprocessors.

## What Changes

- **Codes staged once per chunk.** Each block loads its chunk's K and V
  codes and norms into shared memory with coalesced loads, as the f32
  grouped Kernel stages its rows, instead of every thread reading its own
  key's codes and every value coordinate walking the chunk's codes in
  global memory.
- **Every thread in the value sums**, and the queries' shared memory
  sized to the group, so more blocks fit on a multiprocessor.
- **Split-K.** A block walks several chunks with an online softmax and
  writes one partial; the number of blocks per KV head is set from the
  multiprocessor count and the history (the RFC's occupancy and
  granularity bounds), so the query rotation runs once per block and the
  partials no longer grow with the history.

## Capabilities

### New Capabilities

- `turboquant-decode-attention-throughput`: how the CUDA Provider's decode
  attention over TurboQuant stores reads codes and splits the history.

### Modified Capabilities

None: the results are those `turboquant-kv-history` and
`keep-turboquant-kv-head-exact` specify, to within f32 rounding.

## Impact

- `providers/cuda`: the code split Kernels, their launch, the attention
  scratch; conformance and `decode_attention_times`.
- `magnetar-runtime`: none.

## Non-goals

- The f32 decode Kernels, whose partials also grow with the history (a
  later change if the measurements call for it).
- QJL residual keys (RFC section 3.2), evaluated separately.
