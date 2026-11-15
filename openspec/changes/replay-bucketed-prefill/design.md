## Context

`capture-and-replay-cuda-graphs` (C04) records one decode step per request and replays it for the request's later steps. The recording belongs to the request: it references the request's KV stores and the decode graph's intermediates, and a replay first checks that every resource still has the buffer it had at capture time.

The prefill is outside that profile. On an RTX 3060 (runs 41845390146 and later), a short prompt's warm TTFT is about the cost of a per-node step:

| Model | Warm TTFT, short prompt | Replayed decode step |
|---|---|---|
| Qwen2.5-0.5B (5 prompt tokens) | 15.0 ms | 4.4 ms |
| Mistral 7B `Q4_K_M` | 33.8 ms | 17.4 ms |

What prevents a prefill replay today, in the first-native engine:

- `first_native_component_graphs_for_prompt` builds the prefill graph for the prompt's exact length; the graph cache keeps one graph per length.
- `create_prefill_kv_state` sizes the request's KV cache layout to `prompt_token_count + max_new_tokens`; the in-place stores are allocated at the first `kv-write` and released with the request (`release_request_state`).
- The prefill runs at position 0, once per request, and materializes only its last row's logits (`MaterializedEdges::LastRow`), the LM head reading only that row.

## Goals / Non-Goals

Goals: replay a short prompt's prefill from a recorded CUDA graph when an earlier request of the same shape recorded one. Preserve the existing step capture contract, trust and Memory Manager admission. Keep the logits and tokens identical to the unpadded per-node prefill, and every ineligible prefill on today's per-node path.

Non-goals: see proposal.md. They are chunked prefill, cross-request decode recordings, multi-device capture, and shared captures across concurrent requests.

## Decisions

### Buckets and padding

- **Buckets.** The default bucket list is every multiple of 8 up to 64, then 96 and 128 tokens, configurable with `MAGNETAR_PREFILL_BUCKETS`. The packed CUDA matmuls read their weights once per 8 rows, so padding to the next multiple of 8 adds no weight pass. The first list (16, 32, 64, 128) cost Mistral 7B `Q4_K_M` 88.5 ms against 33.7 ms per node for a 6-token prompt (see Evidence). A prompt of `n` tokens takes the smallest bucket `b >= n`. A prompt longer than the largest bucket runs today's exact-length prefill, unbucketed and per node. Buckets longer than the model's context length are dropped, and a model split by tensor parallelism is not bucketed.
- **Padding.** The bucket's last `b - n` ids repeat the prompt's last token. Any in-vocabulary id is valid: no real row reads a padding row, under causal attention with key count `b` at position 0. The padding rows' K and V are written at rows `n .. b`. These rows lie beyond the committed length, which stays `n` (`PrefillCompleted { tokens: n }`). The first decode step writes row `n` and attends over `n + 1` keys, so the stale rows are never read and are overwritten as decoding proceeds.
- **Equivalence.** Each real row's outputs are a function of rows `0..=i` only, so they equal the unpadded prefill's, up to summation order. The prefill kernels' order depends on the row count: tiled and Tensor Core paths start at 64 rows. Qualification therefore compares logits within the existing f32 tolerance, and greedy tokens exactly, rather than bit-identity. Bit-identity with an unpadded prefill of the same bucket length is required.

### Logits row selection

- **Today.** The prefill's LM head is rewritten (`qualify-fusion-and-optimized-attention`) to read the last row of its input.
- **Bucketed prefill.** The Runtime instead inserts an `embedding` node that gathers row `n - 1` of the final hidden states. Its index is a new Runtime-provided graph input, `input.logits_row` (shape `[1]`). The LM head reads that one row, and the logits keep one row.
- **Fallback.** A graph whose `logits` is not produced by a plain or bias matmul keeps every row's logits; the engine then reads row `n - 1`. Such a prefill is bucketed but cannot be captured.
- **Why `embedding`.** It is an existing portable Operator with a CUDA Kernel and a Reference CPU kernel, so no Operator, WIT or Component change is needed. A replay writes `input.logits_row` like it writes the token ids.

### KV store pool

- **Capacity buckets.** A bucketed request's KV capacity is rounded up to a capacity bucket: the next power of two at or above `b + max_new_tokens`, with a minimum of 256 rows, and at most the model's context length when that holds the request. Requests of similar sizes therefore share store shapes.
- **Naming and ownership.** A pooled set's stores are named after the set (`kv.pool{set}.layer{l}.{k|v}.store`), not after a request's KV cache, and admitted in the Memory Manager under the `kv-store-pool` owner, so the memory they hold stays visible and bounded.
- **Release.** When a bucketed request is released, its set goes back to the per-Model-Instance pool instead of being freed. A failed prefill also leaves its set's stores with the pool.
- **Reuse.** A new bucketed request takes the most recently used free set of its capacity bucket. Its stores keep their Memory Manager admissions and their Device buffers, which is what lets its recordings replay. They are not zeroed again.
- **Bounds.** The pool holds at most `MAGNETAR_KV_POOL_BYTES` (default 1 GiB) and at most 4 store sets per capacity bucket, free or in use. It drops least recently used free sets first. A set larger than the whole budget is not pooled: that request owns its stores. Dropping a set releases its stores; recordings are released with their request beforehand.
- **Admission pressure.** A generation step refused Memory Manager admission is retried once, after the pool's free sets are released. The Memory Manager itself is unchanged.
- **Isolation.** A pooled set holds no state that a later request can read: the request's valid row count starts at 0, and no row is read before it is written. It is never shared by two live requests.

### Prefill recordings

- **Key.** A recording's key is (Model Instance, graph fingerprint, prompt bucket, capacity bucket, store set id). It lives with its store set and is released with it.
- **Capture sequence.**
  1. The first bucketed prefill for a key runs per node. It allocates every bucket-shaped intermediate once, as the first decode step does for decode.
  2. The second one runs inside `begin_step_capture` / `end_step_capture`. The advancing attributes are the decode ones (`position_offset`, `key_count`, `row_offset`); a prefill replays with offset 0.
  3. Later prefills replay it. A replay writes `input.token_ids` (`b` padded ids) and `input.logits_row`, then calls `replay_step(id, 0, inputs)` and downloads one logits row (or reads the Device greedy selection).
- **Intermediates.** A recordable prefill writes its node outputs under the bucket's namespace (`prefill{b}.edge.*`, owned by `prefill{b}`), never under the decode graph's `edge.*` resources, and keeps them after their last reader. A decode step therefore never reallocates a buffer a prefill recording references. Store sets of one bucket share these intermediates; requests run one at a time on a loaded model. They hold about `b` rows of every intermediate edge. All bucket namespaces together hold at most `MAGNETAR_PREFILL_REPLAY_BYTES` (default 1 GiB). Beyond it, the least recently used bucket's recordings and intermediates are released, never those of the bucket in use; that bucket then starts its sequence again.
- **Inputs.** A replay writes the padded ids and the logits row into the bucket's own input resources (`prefill{b}.input.token_ids`, `prefill{b}.input.logits_row`).
- **Failure.** A failed capture or an invalidated replay falls back to the per-node prefill, as decode does. A failed capture declines further captures for that key, whose prefills then run as unrecorded ones. An invalidated replay drops the recording; the next prefill of the key runs per node and the one after captures again. Dropping a pooled store set releases its recordings.

### CUDA Provider

- **Captured kernels.** A prefill's Kernels must run inside a capture without allocating, synchronizing or downloading. This covers the tiled and Tensor Core prefill matmuls and attention, the RMSNorm, RoPE, gated MLP and `kv-write` multi-row paths, and the Device token check of `embedding`.
- **Approach.** The capture already fails on any of these, so the first task is to run the bucketed prefill under capture and fix each Kernel that fails. The expected candidates are split-K workspaces sized per call and host-side shape checks.
- **Padding rows are skipped.** A prefill capture advances one attribute, `padding_rows`, and its replay offset is the bucket's padding row count `b - n`. The packed row matmuls (`q8`, `q4k`, `q5k`, `q6k`, low-bit and i-quant, plain and gated, f32 and int8 activations) read it from the Device and compute only the first `b - offset` rows. The capture allows at most `b - 1`. Positions, key counts and KV rows do not advance in a prefill replay. Skipped rows keep stale but finite values: no real row reads them under causal attention, the logits row gather reads row `n - 1`, and decoding overwrites their KV rows. Other Kernels still compute every row.
- **Attention.** The Tensor Core and tiled attention Kernels over several query rows were excluded from captures. They are now captured; having no replay offset, a capture holding one bounds its replay offset to 0, which is a prefill's.
- **Attention bounds.** Captured block attention sizes its shared memory for the store capacity, which the capacity bucket fixes.

### Controls and reporting

- **Disable controls.** `ProductionLoadedModel::set_prefill_replay(false)` and `MAGNETAR_PREFILL_REPLAY=0` disable bucketing, the pool and prefill capture together. Every prefill then runs today's exact-length per-node path; this is the qualification's before/after switch.
- **Reporting.** The phase ledger reports `prefill_replay` time and the step trace marks replayed prefills. The qualification reports the pool's held bytes.

## Evidence

GPU CI run 43717628081 (RTX 3060), replay against its disable control (`MAGNETAR_PREFILL_REPLAY=0`) in the same run. Greedy tokens are identical in every pair.

| Warm TTFT | Bucket | Replayed | Control (exact length, per node) |
|---|---|---|---|
| Qwen2.5-0.5B, 5 tokens | 16 | 17.1 ms | 14.7 ms |
| Qwen2.5-0.5B, 12 tokens | 16 | 17.0 ms | 18.4 ms |
| Qwen2.5-0.5B, 40 tokens | 64 | 21.8 ms | 40.2 ms |
| Mistral 7B `Q4_K_M`, 6 tokens | 16 | 88.5 ms | 33.7 ms |

- Every bucketed prefill was captured on its second request and replayed afterwards (`[step capture] captured: prefill of 16 rows`, `... 64 rows`).
- The pooled KV stores hold 6.0 MiB on Qwen2.5-0.5B (one 256-row set).
- Padding is not free. The packed CUDA matmuls (`q4k_matmul_rows_kernel` and its twins) process `a` in groups of 8 rows (`Q8_MATMUL_ROWS`) and read the weights once per group. A 6-token prompt takes one pass over the weights; padded to 16 rows it takes two. On a 7B model that costs more than the per-node host work replay removes.
- The 40-token prompt also changes kernels: at 64 rows the prefill matmuls take the Tensor Core path, which starts at 64 rows. Its gain is not replay alone.
- Before this change, run 43693053049, which buckets per node without replay, gave a 19.2 ms warm TTFT for 5 tokens.

With buckets at every multiple of 8 (runs 43723450794 and 43728214947), tokens are again identical in every pair:

| Warm TTFT | Bucket | Replayed | Control |
|---|---|---|---|
| Qwen2.5-0.5B, 5 tokens | 8 | 10.9 ms | 15.9 ms |
| Qwen2.5-0.5B, 12 tokens | 16 | 17.2 ms | 18.4 ms |
| Qwen2.5-0.5B, 40 tokens | 40 | 38.5 ms | 39.8 ms |
| Mistral 7B `Q4_K_M`, 6 tokens | 8 | 45.1 ms | 33.9 ms |

- **Where Mistral's replayed prefill spends its time.** 44.8 of its 45.1 ms are spent waiting for the Device; the replay itself costs 0.16 ms of host time. The per-node control spends 33.1 ms in all. On a 7B model the per-node host work is already hidden behind the Device, so replay has nothing to remove.
- **Padding rows cost Device time in proportion.** 33.7 ms × 8/6 = 44.9 ms, and 33.7 ms × 16/6 = 89.9 ms (88.5 ms measured with 16-row buckets). With several rows, the packed matmuls are bound by compute, not by the weight reads: every padding row costs as much as a real one. The earlier explanation (one weight pass per 8 rows) was incomplete.
- **The 40-token prompt.** Run 43717628081 padded it to 64 rows and measured 21.8 ms replayed. The prefill matmuls take the Tensor Core path from 64 rows, which is faster than the 40-row path. That is a kernel threshold, independent of replay.

## Risks / Trade-offs

- **VRAM held between requests.** The pool keeps stores admitted after their requests end. The budget, the Memory Manager reclaim and the disable control bound it, and the campaign reports it.
- **Padding work.** A prompt of `n` tokens computes `b` rows, up to about 2x when `n` is just over a bucket boundary. Short prompts are host-bound, so the extra Device work is expected to cost less than the per-node host work it replaces; the campaign must show it.
- **Numerical drift.** Bucketed and unpadded prefills can differ in summation order. Greedy tokens and logit error against the per-node exact-length prefill are the acceptance criteria.
- **Concurrency.** Each live request owns a store set, so concurrent requests use different sets and different recordings. A burst beyond the pool's set count runs per node.

## Migration and rollback

Additive and disableable. With `MAGNETAR_PREFILL_REPLAY=0` the engine runs exactly today's path. No WIT, Component or persisted format changes.

## Open questions

- **Bucket list and capacity rounding.** Should they follow the observed prompt distribution? The defaults are a starting point, to be revisited with the campaign.
- **Decode recordings.** Should a pooled store set also keep its decode recording, so the next request skips its two per-node decode steps? This is deferred, see proposal.md.
