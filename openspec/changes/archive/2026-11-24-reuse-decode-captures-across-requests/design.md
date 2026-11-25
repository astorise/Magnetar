## Context

A decode step is captured and replayed through `ProviderExecutionApi::begin_step_capture` and `replay_step` (`capture-and-replay-cuda-graphs`). A capture records the Device addresses of every resource it reads and writes. A replay is therefore valid only while those resources keep their allocations. A replay writes the token input, then advances `position_offset`, `key_count` and `row_offset` by a non-negative offset from the captured step.

Until now, a request recorded its own decode step. Its first step ran per node and allocated every decode-shaped output under the request's session. Its second step was captured (`STEP_CAPTURE_DECODE_INDEX = 2`), and its recording was released with the request.

`replay-bucketed-prefill` added a KV store pool. A short request's stores are a pooled store set (`kv.pool{set}.layer{l}.{role}.store`), reused at the same addresses by later requests of the same capacity bucket. The bucketed prefill's recording is keyed by (prefill graph, bucket, set), and its intermediates live in a bucket namespace (`prefill{b}`).

## Decisions

### Key a pooled request's decode recording by (decode graph, store set)

When the request's `KvStoreSpec` names a pool set, its decode recording is keyed by the decode graph's semantic fingerprint and the set. Every request served by the set reads and writes the same stores.

### A decode namespace per (set, graph)

The steps run in namespace `decode{set}-{fnv64(graph fingerprint)}`, through the dispatch context's `resource_namespace` as a bucketed prefill does. Their outputs, the captured token input and the logits are owned by `Session(namespace)`. They are therefore kept past the request, at the same addresses. Two sets never share a namespace, so concurrent requests on different sets never write each other's intermediates.

### Sequence: Warm, Capture, Replay

This is the bucketed prefill's sequence (`RecordingStage`, now shared by both):

- **Warm.** The first step for a key runs per node in the namespace, which allocates its outputs.
- **Capture.** The next step is captured.
- **Replay.** Every later step replays with offset `position - recorded position`.

A step before the recorded position cannot replay, since the offset is unsigned. It captures again, replacing the recording. This is safe at any decode index, because the namespace's outputs are already allocated. With prompts of one length, the second request captures once more and every later step replays. With varying lengths, the recording moves to the shortest prompt seen.

### Fallback keeps the previous behavior

A Provider that cannot capture is detected at the Warm stage. The key is declined before anything is kept in a namespace. A failed capture also declines the key, and an invalidated replay drops the recording; the key then captures again at its next step. A declined set's requests run the per-request path unchanged.

### Release with the set

`release_pooled_kv_sets` releases the set's prefill recordings (as before), its decode recordings, and every resource whose identifier starts with `decode{set}-`, before releasing its stores.

## Risks / Trade-offs

- **Device memory.** One decode step's intermediates per pooled set stay resident between requests. For Qwen2.5-0.5B this is about 1 MB of activations, plus a 0.6 MB logits row. They are released with the set, whose budget (`MAGNETAR_KV_POOL_BYTES`) already bounds its stores.
- **Shortest prompt.** A request shorter than every earlier one on its set pays one capture, as every request did before.
