## Why

A TurboQuant KV history (`turboquant-kv-history`) quantizes every row as
it is written. Two costs follow that the history's length does not
justify:

- **Short histories decode slower than f32.** At 512 tokens Qwen2.5-0.5B
  decodes at 0.84-0.86x f32's speed in `tq4` and `tq3` (runs 44203342858,
  44266581438; task 5.5): the query rotation, the offsets and the inverse
  rotation are a fixed cost per step, and a short history saves too few
  bytes to pay for them.
- **The first tokens are quantized too.** Attention keeps a large share of
  its weight on a sequence's first tokens ("attention sinks"); their
  quantization error weighs on every later step.

The hybrid KV cache proposed for this Runtime (an RFC by Gemini, given by
the user) keeps sinks and a recent window in full precision and quantizes
the rest. This change is its first step: the history's first rows stay
exact.

## What Changes

- **An exact head.** A TurboQuant store keeps its first `E` rows exact, in
  f32 (`MAGNETAR_KV_EXACT_ROWS`, 1,024 when unset; `0` turns it off). The
  rows from `E` on are TurboQuant codes as today. The head covers the
  sinks of every history, and the whole of a history of up to `E` rows.
- **Decode by regime.** While a history holds at most `E` rows, decode
  attention reads only the exact head with the f32 Kernels: no TurboQuant
  cost at all. Beyond it, the exact head and the codes are scored
  together and their partials combined.
- **Semantics.** A store row's value is the row itself below `E`, its
  TurboQuant round trip (with the store's offsets) from `E` on, on every
  Provider.
- **Admission and compatibility.** A store is admitted at its codes' size
  plus `E` f32 rows; `E` is part of the KV compatibility and layout
  metadata.

## Capabilities

### New Capabilities

- `turboquant-exact-head`: the exact head of a TurboQuant KV store, its
  configuration, semantics, admission and decode regimes.

### Modified Capabilities

None: a store without an exact head behaves as `turboquant-kv-history`
specifies.

## Impact

- `magnetar-provider-api`: `TurboQuantStoreState` and stored sizes with an
  exact head; the store creation call carries it.
- `magnetar-runtime`: the policy, admission, compatibility, Reference CPU.
- `providers/cpu`: both CPU Providers' stores.
- `providers/cuda`: the head buffer, `kv-write`, decode by regime
  (capture bounds), the dequantized scratch of prefill reads.

## Non-goals

- The recent window of the RFC (a ring buffer of the last `W` rows kept
  exact): the padding rows of bucketed and chunked prefills, which decode
  later writes over, would evict rows still in the window; a second step.
- QJL residual keys and the RFC's split-K heuristic.
