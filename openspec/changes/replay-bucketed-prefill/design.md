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

- **Buckets.** The default bucket list is 16, 32, 64, 128 tokens, configurable with `MAGNETAR_PREFILL_BUCKETS=16,32,64,128`. A prompt of `n` tokens takes the smallest bucket `b >= n`. A prompt longer than the largest bucket runs today's exact-length prefill, unbucketed and per node. Buckets longer than the model's context length are dropped, and a model split by tensor parallelism is not bucketed.
- **Padding.** The bucket's last `b - n` ids repeat the prompt's last token. Any in-vocabulary id is valid: no real row reads a padding row, under causal attention with key count `b` at position 0. The padding rows' K and V are written at rows `n .. b`. These rows lie beyond the committed length, which stays `n` (`PrefillCompleted { tokens: n }`). The first decode step writes row `n` and attends over `n + 1` keys, so the stale rows are never read and are overwritten as decoding proceeds.
- **Equivalence.** Each real row's outputs are a function of rows `0..=i` only, so they equal the unpadded prefill's, up to summation order. The prefill kernels' order depends on the row count: tiled and Tensor Core paths start at 64 rows. Qualification therefore compares logits within the existing f32 tolerance, and greedy tokens exactly, rather than bit-identity. Bit-identity with an unpadded prefill of the same bucket length is required.

### Logits row selection

- **Today.** The prefill's LM head is rewritten (`qualify-fusion-and-optimized-attention`) to read the last row of its input.
- **Bucketed prefill.** The Runtime instead inserts an `embedding` node that gathers row `n - 1` of the final hidden states. Its index is a new Runtime-provided graph input, `input.logits_row` (shape `[1]`). The LM head reads that one row, and the logits keep one row.
- **Fallback.** A graph whose `logits` is not produced by a plain or bias matmul keeps every row's logits; the engine then reads row `n - 1`. Such a prefill is bucketed but cannot be captured.
- **Why `embedding`.** It is an existing portable Operator with a CUDA Kernel and a Reference CPU kernel, so no Operator, WIT or Component change is needed. A replay writes `input.logits_row` like it writes the token ids.

### KV store pool

- **Capacity buckets.** A bucketed request's KV capacity is rounded up to a capacity bucket: the next power of two at or above `b + max_new_tokens`, with a minimum of 256 rows. Requests of similar sizes therefore share store shapes.
- **Release.** When a bucketed request is released, its stores are kept in a per-Model-Instance pool keyed by capacity bucket instead of being freed. They remain admitted in the Memory Manager under a pool owner, so the VRAM they hold stays visible and bounded.
- **Reuse.** A new bucketed request with that capacity bucket takes a pooled store set if one is free. That set's stores keep their addresses, which is what lets its recordings replay.
- **Bounds.** The pool holds at most `MAGNETAR_KV_POOL_BYTES` (default: 10 % of the Device budget used for layer placement) and at most 4 store sets per capacity bucket. It drops least recently used sets first. Under admission pressure, the Memory Manager reclaims pooled sets before failing an admission. Dropping a set releases its recordings.
- **Isolation.** A pooled set holds no state that a later request can read: its valid row count is reset to 0, and no row is read before it is written. It is never shared by two live requests.

### Prefill recordings

- **Key.** A recording's key is (Model Instance, graph fingerprint, prompt bucket, capacity bucket, store set id). It lives with its store set and is released with it.
- **Capture sequence.**
  1. The first bucketed prefill for a key runs per node. It allocates every bucket-shaped intermediate once, as the first decode step does for decode.
  2. The second one runs inside `begin_step_capture` / `end_step_capture`. The advancing attributes are the decode ones (`position_offset`, `key_count`, `row_offset`); a prefill replays with offset 0.
  3. Later prefills replay it. A replay writes `input.token_ids` (`b` padded ids) and `input.logits_row`, then calls `replay_step(id, 0, inputs)` and downloads one logits row (or reads the Device greedy selection).
- **Failure.** A failed capture or an invalidated replay falls back to the per-node prefill, as decode does. A failed capture declines further captures for that key. An invalidated replay drops the recording.

### CUDA Provider

- **Captured kernels.** A prefill's Kernels must run inside a capture without allocating, synchronizing or downloading. This covers the tiled and Tensor Core prefill matmuls and attention, the RMSNorm, RoPE, gated MLP and `kv-write` multi-row paths, and the Device token check of `embedding`.
- **Approach.** The capture already fails on any of these, so the first task is to run the bucketed prefill under capture and fix each Kernel that fails. The expected candidates are split-K workspaces sized per call and host-side shape checks.
- **Attention bounds.** Captured block attention sizes its shared memory for the store capacity, which the capacity bucket fixes.

### Controls and reporting

- **Disable controls.** `ProductionLoadedModel::set_prefill_replay(false)` and `MAGNETAR_PREFILL_REPLAY=0` disable bucketing, the pool and prefill capture together. Every prefill then runs today's exact-length per-node path; this is the qualification's before/after switch.
- **Reporting.** The phase ledger reports `prefill_replay` time and the step trace marks replayed prefills. The qualification reports the pool's held bytes.

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
