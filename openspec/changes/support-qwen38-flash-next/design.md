## Context

### Target model

Qwen3.8-Flash-Next is published as Hugging Face `model_type` `qwen4_exp` under the qwen-community-1.0 license. It has 180B parameters: 125B in the transformer and 51B in an n-gram embedding table. About 6B are active per token.

| Part | Geometry |
| --- | --- |
| Layers | 48: three Gated DeltaNet (GDN, `linear_attention`) then one QSA (`full_attention`), repeated 12 times; hidden 2560 |
| GDN mixer | 16 key heads, 48 value heads, head dim 128, depthwise convolution kernel 4 |
| QSA attention | 24 query heads, 2 KV heads, head dim 256, partial rotary 0.25, interleaved M-RoPE sections [11, 11, 10], rope theta 1e7 |
| QSA indexer | 4 heads, 1 KV head, dim 128, token budget 2048, compression ratio 4 |
| MoE | 512 experts, top-10, plus a gated shared expert; expert intermediate 640 |
| Gated residual | 4 residual streams (`hc_count`), low-rank 320 |
| N-gram PLE | n-gram size 3, base 20M rows, 8 heads per n-gram, 128 shards, dim 2560, added at layer 2 (weights under `layers.1.ple.*`) |
| Other | vocabulary 248,320; one MTP layer; a vision tower (out of scope) |

The experts hold about 120.8B parameters: 242 GB in BF16, ~68 GB at 4.5 bits per weight, ~38 GB at 2.5 bits. The always-active part is about 4.2B parameters. The n-gram table has 51B parameters, read sparsely: a few rows per token.

### Reference: Strata on the GPU runner

Run 45446491115 on `arc-gpu-magnetar` (2 × RTX 3060 12 GB, i7-7700, 31 GB RAM, NVMe) used Strata d46f5f2 and ISTA-DASLab IQ2_XS (39.2 GB model, 28.8 GB n-gram table). Context was 8192 with an int8 KV cache.

How Strata placed the model:
- Layers are split across both GPUs.
- Experts are rewritten once into a 33 GiB file and mapped from the SSD. The page cache holds about 24 GB; the process itself uses 7–9 GB.
- The GPU expert cache holds 6,483 of 24,576 experts in 8.57 GiB, with 87–98 % hits once warm.
- The n-gram table stays on the SSD.

Results, as median and range, with no drafting:

| Measure | Result |
| --- | --- |
| Decode | 23.3 tokens/s (16.1–34.1) |
| TTFT, ~46-token prompt | 1.15 s |
| Prefill, 4,019 tokens | 353 tokens/s |

With DFlash2 drafting, decode reaches 38.3 tokens/s. The no-drafting column is Magnetar's target; `benchmark/README.md` has the details.

### What Magnetar already has

- **Graph contract.** `TensorEdge.recurrent_state` carries `GraphRecurrentStateMetadata {resource_id, layer, slot, behavior: Input|Output, compatibility_key}`. `ExecutionGraph::recurrent_state_bindings()` validates exactly one consumed input and one produced output per slot, with the same shape. The Runtime checks that prefill, decode and segment graphs agree on schemas (`validate_recurrent_graph_pair`, `validate_segment_recurrent_scope`).
- **WIT 1.11.0.** `recurrent-resource(layer, slot, shape)`, `recurrent-input(resource-id)` (a new sequence starts from zeros) and `recurrent-output(resource-id, edge-id)` (commit with the accepted token, atomically with KV; a failure preserves the previous generation).
- **Operator.** `gated-delta-mixer` takes 13 inputs and produces 3 outputs: the activation, the next convolution state and the next recurrent state. It has f32 Reference semantics in magnetar-provider-api, checked against Transformers fixtures (synthetic tensors, and real Qwen3.5-0.8B-Base first-mixer weights). The first-native dispatcher already runs it, with state passed as ordinary tensors.
- **Refusals.** `plan_execution_graph`, `prepared_plans.rs` and `execute_first_native_graph_nodes` all refuse any graph with a recurrent edge.
- **Reused pieces.** MoE routing and experts (`execute-and-cache-moe-experts`), packed i-quants (IQ2/IQ3), layer segments across two GPUs, and budgeted weight residency (`operate-budgeted-weight-residency`, `enforce-local-multigpu-and-expert-placement`).

## Goals / Non-Goals

**Goals**
- Lot 1: recurrent graphs run in the first-native Runtime with the WIT semantics, on Reference CPU, cpu-native and CUDA.
- Lot 2: Qwen3.5-0.8B-Base (Apache-2.0) generates the same greedy tokens as Transformers. It is the dense, small proof of the GDN hybrid.
- Lots 3–6: Qwen3.8-Flash-Next runs on `arc-gpu-magnetar` within reach of Strata's no-drafting numbers.

**Non-Goals**
- The vision tower and multimodal input.
- Continuous batching, prefix reuse and speculative verification for recurrent graphs before their own lots. Until then they are refused or bypassed explicitly.
- BF16 checkpoints of the full model on the runner: 242 GB of experts do not fit.

## Decisions

### D1. Recurrent state is a session resource committed with KV

The state belongs to the step's KV cache. `KvCache` holds, per graph resource id, the committed value and the value pending in the current attempt (`RecurrentStateResourceBinding`: resource, allocation, shape, compatibility key), admitted under the session's memory owner.

**Execution**
- The executor binds each input edge to the pending value, else the committed value, else a zero tensor of the layout's shape. A caller-supplied state input is refused.
- After a successful step, each output edge's value is copied under an attempt-unique resource and becomes the pending value, replacing (and releasing) the previous one. A failed step stages nothing.
- `KvUpdateTransaction::commit` adopts the pending values once every KV layer is promoted and releases the values they replace.
- Discarding the attempt's pending KV releases its pending state. Releasing the request's state, or the cache, releases everything.

Rejected alternative: copy the output over the input in place. A failed or partial step would then corrupt the only copy of the state, which the WIT forbids.

### D2. Passes, padding and commit

- Passes of one attempt (a prefill run in several exact-length passes) read the attempt's pending state.
- The state is committed with the accepted token, as KV is, so the first decode step reads the prefill's state.
- Prefill buckets and the chunk graph right-pad the prompt. Padding rows would advance the state, so a model whose graphs carry recurrent state prefills at the prompt's exact length: buckets and chunking are turned off for it.
- Layer segments do not commit through the KV transaction; they refuse recurrent graphs until segment commits carry the state. Each slot would then stay on its segment's device; the scope check already exists.

### D3. Explicit refusals until each feature has its lot

A graph with recurrent bindings:
- is not captured for step replay, in prefill or decode. The trace (`MAGNETAR_STEP_CAPTURE_TRACE=1`) gives the reason, until replay alternates paired committed/pending buffers (lot 1b).
- does not reuse a prefix: state cannot be rebuilt from a KV prefix. The first-native engine has no cross-request prefix reuse yet; when it lands, it bypasses recurrent models with an observation.
- is refused by continuous batching and speculative verification, which the first-native engine does not run yet either.

No path silently runs with stale or shared state.

### D4. Kernels

- The Reference CPU semantics remain the oracle.
- cpu-native gets a decode form (one token: update the recurrent state, read the output) and a chunked prefill form.
- CUDA gets the same two forms: the decode form is one block per value head, the prefill form is a chunked scan.
- Both are qualified against the Reference within tolerance, including state outputs, before any advertisement.

### D5. Qwen3.5 graph in the Qwen Component

- The Component recognizes the Qwen3.5 text configuration (`layer_types`, `linear_*` geometry, gated full attention with partial rotary).
- It emits, per layer, either a `gated-delta-mixer` node with two recurrent slots (`conv`, `delta`) or a gated attention block with KV.
- Operators missing from the catalog for the attention block are added additively with Reference semantics. Norm and gate formulas are copied from the pinned Transformers source, not assumed.

### D6. Flash-Next blocks (lots 3–6)

- QSA, the four-stream gated residual and the n-gram PLE are new portable operators. Their semantics come from a pinned upstream modeling source and are checked against fixtures generated from it, as for Qwen3.5. Until that source is pinned, these lots stay design-only.
- The n-gram table is a disk-resident weight. The residency manager reads only the hashed rows each step needs; it is never fully admitted.
- Experts reuse the MoE operator and are placed by the residency policy:
  - mapped or in RAM;
  - an adaptive GPU cache sized to free VRAM;
  - misses computed by the CPU Provider in place, rather than uploaded.

## Risks / Trade-offs

- **QSA and residual semantics.** The upstream semantics may not be public in a reviewable form. Mitigation: those lots stay blocked rather than reverse-engineered from Strata's binary engine.
- **Lot 1 replay cost.** Disabling step replay for recurrent graphs costs decode speed. Mitigation: lot 1b restores it with paired buffers.
- **Mapped experts on a slow SSD.** Misses can dominate decode. Mitigation: measure hit rate and miss latency in the same way as Strata's logs.
- **License.** The qwen-community-1.0 license must be reviewed before CI downloads the checkpoint. No weights are committed.

## Open Questions

- The exact QSA indexer selection and when it reduces to dense attention (context ≤ budget × ratio?).
- Whether the n-gram hash and shard layout is stable across the GGUF and Hugging Face exports.
- Which GGUF tensor types ISTA-DASLab uses beyond IQ2_XS/IQ3_XXS/Q2_0, and whether they all have Magnetar kernels.
