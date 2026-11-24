## Why

After the per-node host cuts recorded in `attribute-redacted-inference-costs`, a Qwen2.5-0.5B decode step on an RTX 3060 (GPU run 41221305032) spends about 2.4 ms of its 14.9 ms on KV history bookkeeping. For each of the 24 layers and both roles, every step:

1. dispatches a `concat` that copies the whole committed history plus the new row into a fresh buffer (`kv_append`, 1.97 ms for 48 dispatches);
2. copies that buffer again into a fresh committed generation at commit (`commit`, 0.48 ms).

Both copies grow with the history, and the history's shape changes every step. That rules out replaying a captured step (`capture-and-replay-cuda-graphs`, C04).

## What Changes

- New Runtime Operator `kv-write`. Inputs: a KV store `[capacity, width]` and new rows `[r, width]`. Attribute: `row_offset`. It writes the rows in place at `row_offset`, and its output is the store itself (`supports_in_place`).
- `attention` gains an optional `key_count` attribute: attend to the first `key_count` key/value rows only. When it is absent, every row is attended, as before.
- When the executing Provider advertises a `kv-write` Kernel and an `attention` Kernel with the `key-count` feature, the first-native engine keeps each layer and role's history in one store per request. Its capacity is the request's prompt plus `max_new_tokens` rows.
  - Prefill writes the prompt rows at offset 0.
  - Each decode step writes its new row at the committed length and runs attention with `key_count` set to the committed length plus the new rows.
  - Commit advances the committed length, with no copy.
  - A discarded or failed step leaves the committed length, and so every committed row, unchanged.
- Otherwise (another Provider, a multi-device segment, a request without a known capacity), the current concat and copy path is used unchanged.

## Capabilities

### Modified Capabilities

- `kv-cache`: in-place history append for first-native generation.

## Impact

- `magnetar-runtime`: operator catalog, Reference CPU, first-native graph executor, KV commit, discard and release.
- `providers/cpu` and `providers/cuda`: `kv-write` Kernels, and `key_count` in attention, behind the advertised `key-count` feature.

There is no WIT change: the Operator and attribute are Runtime- and Provider-internal, and Components never see them. Providers that advertise neither keep working through the existing path.

## Non-goals

- Paged or shared KV storage (`operate-paged-kv-storage`) and prefix reuse across requests.
- CUDA graph capture itself (C04), which this change prepares by making the decode step's shapes fixed.
