## Why

A same-machine comparison on the GPU runner (benchmark run 44490170115) measured one request of `The capital of France is` (5 prompt tokens) with 16 greedy new tokens on Qwen2.5-0.5B-Instruct:

| Engine | Request (median) | Decode, per token |
| --- | --- | --- |
| Magnetar (CUDA) | 92.6 ms | 4.1 ms once replayed |
| llama-server (FP16 GGUF) | 80.1 ms | 4.86 ms |
| Ollama (FP16 GGUF) | 79.4 ms | 4.55 ms |

Magnetar's replayed decode step is the fastest of the three, yet its request is the slowest. Every measured request paid two slow decode steps, at 11.7-16.7 ms each. These are the first decode step, run per node, and the second, run per node inside a capture (`capture-and-replay-cuda-graphs`). A decode recording is keyed by the request and released with it, so every request captures again.

The bucketed prefill (`replay-bucketed-prefill`) already avoids this. Its recording is keyed by its bucket and its pooled KV store set, which outlive the request, and its intermediates live in a bucket namespace. Its first request runs per node, its second captures, and every later one replays.

## What Changes

- **Pooled decode recordings.** A request whose KV stores are a pooled store set keys its decode recording by the decode graph and the set, instead of by the request.
- **Decode namespace.** Its decode steps run in the set's decode namespace (`decode{set}-{graph}`), whose intermediates, token input and logits outlive the request. The first request runs its first decode step per node and captures its second. Every later step of every request on the set replays.
- **Earlier positions.** A replay only advances positions. A step earlier than the recorded one (a shorter prompt) therefore captures again. The intermediates are in place, so it captures at once.
- **Fallback.** A failed capture, or a Provider that cannot capture, declines the key. That set's requests then capture their own steps, as before.
- **Release.** When the pool drops a set, the set's decode recordings and namespace are released with its stores.

## Capabilities

### New Capabilities

- `pooled-decode-replay`: decode recordings keyed by graph and pooled store set, the decode namespace, capture again at an earlier position, fallback, release with the set.

### Modified Capabilities

None. The step contract of `capture-and-replay-cuda-graphs` is unchanged, and so are the KV store pool and the bucketed prefill of `replay-bucketed-prefill`.

## Impact

- `magnetar-runtime` (first-native engine): `StepReplayState::pooled_decodes`, `execute_pooled_decode_step`, and the set release in `release_pooled_kv_sets`.
- No Provider, WIT or Component change.
- Requests with unpooled stores (longer prompts, or a pool that cannot hold the set) are unchanged.
