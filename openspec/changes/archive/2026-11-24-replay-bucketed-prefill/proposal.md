## Why

Decode steps already replay a captured CUDA graph (`capture-and-replay-cuda-graphs`): a Qwen2.5-0.5B decode step on an RTX 3060 went from 12.8 ms per node to 4.4-8.0 ms replayed, because about 9.5 ms of a per-node step is host work (dispatch, resolution, admission) identical from one step to the next.

The prefill still runs per node, and for short prompts it is that same host work: warm TTFT is 15 ms at 5 prompt tokens for Qwen2.5-0.5B (run 41845390146), and 34 ms for Mistral 7B `Q4_K_M` against a 17 ms replayed decode step. Long prompts are bound by the Device (TTFT 96 ms at 512 tokens, 715 ms at 4,096), so host work matters only for short ones, which are the common case for a serving workload with many requests.

A prefill cannot be replayed today, because nothing about it repeats:

- its graph is built for the prompt's exact length;
- the request's KV stores are allocated for it, with a capacity of its prompt plus `max_new_tokens`, and freed with it;
- it runs once per request.

A capture is only ever replayed against the exact resources it recorded, so a captured prefill could never be replayed.

## What Changes

- **Prompt-length buckets.** A short prompt runs as a bucket-length prefill (for example 16, 32, 64, 128 tokens), right-padded. Causal attention keeps every real row independent of the padding. The padding rows' K and V land beyond the committed length, like a replayed decode step's do, and the first decode step overwrites them. The committed prompt length stays the real one.
- **Logits row selection.** The Runtime selects the last real row, not the last row, for the LM head. It does so with an `embedding` (row gather) node whose row index is a Runtime-provided graph input, so a replay only writes that input. This needs no WIT or Component change: the Runtime already rewrites the prefill LM head to read one row.
- **A KV store pool.** For a Model Instance, the Runtime keeps released KV stores in a bounded pool keyed by their capacity bucket. A later request with the same capacity bucket reuses them at the same addresses. The pool is admitted and accounted in the Memory Manager, bounded by a budget, and drained under memory pressure.
- **Prefill capture and replay.** A bucketed prefill runs per node the first time for its (bucket, capacity, store set) key, then inside a capture the second time. Later prefills with that key replay it through the existing step contract: they write the token ids and the logits row index, with step offset 0.
- **Controls.** The feature is disabled by `ProductionLoadedModel::set_prefill_replay(false)` and `MAGNETAR_PREFILL_REPLAY=0`; the bucket list and the pool budget are configurable. Every ineligible prefill (a long prompt, an unsupported Provider, an invalidated capture) runs per node, unchanged.
- **Qualification.** Warm TTFT for short prompts on Qwen2.5-0.5B and Mistral 7B, replay against per node in the same run. Logits and tokens must match the unpadded per-node prefill. VRAM held by the pool is reported.

Status: **Proposed.** No code changes with this proposal.

## Capabilities

### New Capabilities

- `bucketed-prefill-replay`: prompt-length buckets with right padding, logits row selection, a pooled KV store reuse, prefill capture and replay keys, controls and qualification.

### Modified Capabilities

None. The step capture contract of `capture-and-replay-cuda-graphs` is reused unchanged: a prefill is captured and replayed as a step with offset 0.

## Impact

- `magnetar-runtime` (first-native engine): bucketing, padding, the logits row graph rewrite, the KV store pool and its Memory Manager accounting, the prefill recording table, controls and reporting.
- `providers/cuda`: none expected beyond what decode capture already supports. The bucketed prefill's Kernels must be capturable: tiled and Tensor Core prefill attention and matmuls without host synchronization or allocation during capture. This is to be confirmed, and fixed where needed.
- No WIT change; Components are unaffected.

## Dependencies

- `capture-and-replay-cuda-graphs` (step capture contract, Device-side offset, replay validation).
- `append-kv-history-in-place` (in-place stores, `key_count`, rows beyond the committed length).
- `qualify-fusion-and-optimized-attention` (prefill LM head on one row).

## Non-goals

- Chunked prefill of long prompts. It could reuse the same machinery later with a fixed chunk size and an advancing offset, but long prompts are bound by the Device.
- Replaying decode recordings across requests: a pooled store set could also keep its decode recording, considered after this change is measured.
- Multi-device and tensor-parallel prefill capture.
- Concurrent requests sharing one capture: each pooled store set has its own recordings.
