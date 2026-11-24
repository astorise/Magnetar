## Context

First-native KV history today is a chain of immutable tensors:
- Each decode step builds a new `[n + 1, width]` tensor with `concat`, which becomes the step's pending resource (`kv.{cache}.layer{L}.{role}.pending`).
- Commit copies the pending resource to a new committed generation (`...gen{N}`) and releases the previous generation.

Keeping each committed generation separate made rollback trivial (Correctif 11): a failed multi-layer commit never touched the previous history. The cost is two full-history copies per layer, per role and per step.

## Decisions

### One store per layer and role, written beyond the committed length

A request's KV capacity is known when its cache is created: `KvCacheLayoutMetadata` already records `prompt + max_new_tokens` positions. Each layer and role gets one store `kv.{cache}.layer{L}.{role}.store` of `[capacity, width]`, admitted once, plus a committed length `n` kept in `FirstNativeLayerKvState`.

A step writes its rows at offset `n`, which is never inside the committed prefix:
- Committing sets `n` to `n + r`.
- Aborting or discarding leaves `n` unchanged. The rows written beyond `n` are simply overwritten by the next step.

Rollback therefore needs no copy and keeps the same guarantee: committed rows are never modified by a step that does not commit.

### `kv-write` Operator

`kv-write(store [C, W], rows [r, W]; row_offset) -> store`. Its properties:
- Its memory behavior declares `supports_in_place`, and its output resource is the store itself.
- Providers implement it as a row copy into the existing allocation, with no new allocation.
- `row_offset + r > C` is rejected (`ShapeUnsupported`), so a capacity overrun can never write past the store.

### `attention` `key_count`

The Operator stays `attention(q, k, v)`. Today `kv_seq_len` comes from K's row count. With `key_count = m`, every Provider treats the first `m` rows of K and V as the whole key sequence, including the causal offset `m - seq_len`, and must not read the rows beyond them.

Attribute validation is the Runtime's, so a Provider that ignored the attribute would silently attend to unwritten rows. The Runtime therefore sets `key_count` only for Kernels whose advertisement lists the `key-count` feature.

### Gating

The in-place path is used for a request when all of the following hold:
- the executing Provider has an active prepared `kv-write` Kernel;
- its `attention` Kernel advertises `key-count`;
- the graph execution is a whole-model first-native step, not a multi-device segment;
- the cache layout records a capacity.

The decision is made once per request, when its KV state is created, and it holds for every step. Any other request uses the existing concat path. Both paths stay covered by tests.

### Memory

A store is admitted once, at its full capacity: `capacity * width * 4` bytes per layer and role, at most `prompt + max_new_tokens` rows. That is up to twice what the concat path holds at its peak, with nothing copied per step. Finished or failed requests release stores through `release_pending_kv_resource` and the request-state release, which are extended to stores.

A new store starts as zeros. `ProviderExecutionApi::write_zeroed_tensor` lets the Provider set them where the store lives; the CUDA Provider zeroes the allocation on the Device. The default declines, and the Runtime then writes a host tensor of zeros as before. Either way, every value is zero.

## Risks / Trade-offs

- **Unwritten rows.** A Provider bug that reads beyond `key_count` would read rows written by a failed step or never written. The conformance tests fill the store's tail with NaN and require the attention result to stay identical.
- **Longer sessions.** Capacity is fixed per request. A future multi-turn cache that grows beyond its capacity must fall back to the concat path or reallocate. No current first-native path reuses a KV cache across requests.

## Evidence

Real Qwen2.5-0.5B, greedy, on an RTX 3060 (GPU CI):
- **Baseline: run 41607144703** (copy-based history, after the per-node cuts).
- **In place: run 41611795148** (Runtime 633f4b7, CUDA Provider 03485c0). The new device conformance passes: in-place `kv-write`, plus bounded attention over a NaN-filled store tail on both attention kernels.

| | Copy-based (41607144703) | In place (41611795148) |
|---|---|---|
| Max logit diff vs. independent corpus | 1.3e-5 | 1.3e-5, same tokens |
| KV append (48 per step) | 1.90 ms | 1.30 ms (`kv-write`) |
| KV commit | 0.47 ms | 0.03 ms |
| Decode step (profile) | 14.4 ms | 13.3 ms |
| Inter-token latency slope | about 0.01 ms/token | none measurable (-0.02 ms/token, buckets 12.8-14.5 ms) |
| Campaign TPOT median (5 x 128 tokens) | 14.7 ms | 13.4 ms |
| Warm TTFT median | 48.4 ms | 44.0 ms |
| Decode throughput | 68.0 tok/s | 74.7 tok/s |

Reference CPU on the same checkpoint: logits unchanged (5.4e-5, same tokens), commit 0.47 -> 0.10 ms. Its `kv-write` writes a copy of the store, so its KV append does not improve; `providers/cpu` and CUDA write in place.

Device-zeroed stores: GPU CI run 41835571121 (Runtime d20a619, CUDA Provider 100e5ba), against run 41829321682:
- The stores are no longer built on the host or uploaded.
- CUDA logits are unchanged (1.3e-5, same tokens), and 13 of 16 steps still replay.
- No gain is measurable on the campaign's prompt. Each of its 48 stores holds about 68 KB, and the warm TTFT median is 15.6 ms against 15.3 ms, within run-to-run spread. The prefill's KV append fell from 2.4 to 1.6 ms, but every prefill phase fell by about as much between those two runs, so that drop does not isolate the change.
- What is removed grows with `prompt + max_new_tokens`: at 4096 + 512 rows, it is about 113 MB per request. The campaign does not run contexts that long yet.
