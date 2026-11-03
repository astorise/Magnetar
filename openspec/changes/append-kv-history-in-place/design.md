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

## Risks / Trade-offs

- **Unwritten rows.** A Provider bug that reads beyond `key_count` would read rows written by a failed step or never written. The conformance tests fill the store's tail with NaN and require the attention result to stay identical.
- **Longer sessions.** Capacity is fixed per request. A future multi-turn cache that grows beyond its capacity must fall back to the concat path or reallocate. No current first-native path reuses a KV cache across requests.
