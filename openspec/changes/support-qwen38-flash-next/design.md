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

- QSA, the four-stream gated residual and the n-gram PLE are new portable operators. Their semantics come from the pinned upstream modeling source of D7 and are checked against fixtures generated from it, as for Qwen3.5.
- The n-gram table is a disk-resident weight. The residency manager reads only the hashed rows each step needs; it is never fully admitted.
- Experts reuse the MoE operator and are placed by the residency policy:
  - mapped or in RAM;
  - an adaptive GPU cache sized to free VRAM;
  - misses computed by the CPU Provider in place, rather than uploaded.

### D8. The n-gram table is read by row and never admitted (task 4.4)

Choices here follow performance: a decode step must not wait on the disk.

**What the code does today.** Three things stop a table of about 320 million rows from running:
- Model Loading reads each tensor whole into a `HostTensor` (`stream_stage_tensor`) and stages it as a Provider resource. Peak host memory is one tensor, which for this table is over 100 GB in `f32`.
- `ngram-embedding` takes the table as a dense input, and its Reference semantics cap every tensor at 67,108,864 elements.
- The payload source is only borrowed (`&dyn ProductionArtifactPayloadSource`) during the load, so nothing could read a row later.

**What a step costs.** Every layer with an n-gram embedding reads `heads` rows per token: with 16 to 48 heads and tens of such layers, one decode token reads on the order of 1,000 to 1,700 random rows of a few hundred bytes. The Strata reference decodes at 23 tokens/s, 43 ms per token, so the lookup has to hide behind compute. Measured on this development VM's virtual disk with a 1 GiB file of 256-byte rows (indicative only: a real NVMe is faster, the ranking is what matters), cold cache, 1,728 rows:

| Access | Time |
|---|---|
| `pread` one row at a time | 158 ms |
| `pread` on 16 threads | 77 ms |
| `mmap`, page faults one at a time | 238 ms |
| `mmap`, one `madvise(WILLNEED)` per row, then read | **13 ms** |
| any of them, warm (page cache) | 0.2 to 1 ms |

A batched `WILLNEED` queues every read at once, so the kernel keeps the device's queue full without any threads of ours; it is 6 times faster than the best threaded `pread` and 18 times faster than faulting page by page. Hot n-grams stay in the page cache, which makes a Runtime row cache pointless until a measurement says otherwise.

**Options.**
- *A. A row-addressed resource in every Provider* (a new `TensorValue` or resource kind). Rejected: the largest blast radius for a weight read a few dozen times per token, and each Provider would reimplement the I/O.
- *B. A Runtime-side gather over a memory-mapped table (chosen).* The table stays in its artifact. The Runtime keeps a `RowTable` per Model Instance and uploads nothing.
- *C. Prefetching the whole table into RAM.* Rejected: it does not fit.

**Design.**
- *Mapped access.* The payload source gains `map_payload`, returning a `MappedPayload`: a read-only mapping of a tensor's bytes that the *source* owns (the Runtime sees no path, handle or pointer, and has no `unsafe`; the one `unsafe`, `memmap2`'s mapping, stays in the loader). `Ok(None)` means the source cannot map, and the table is staged whole as before. A table made of several shards (`stack_ngram_shards`) is one mapping to the Runtime: the loader concatenates its shards' mappings. A row is read by copying its bytes out of the mapping and converting them to `f32`; there is no intermediate buffer.
- *One prefetch per step.* When a step starts, its token ids are a host input and the layers' context states are bound (zeros on a new sequence, else the pending or committed state). The executor computes the row ids of every layer's `ngram-embedding` from them (`NgramEmbedding::row_ids`, the operator's own shared hash, so there is one definition) and hints all of them at once, sorted and merged by page, before the first node runs. The reads then overlap the compute of the layers before the first embedding, and each node finds its rows resident. Every layer's context is the same `ngram_size - 1` tokens, so the executor reads one layer's state, not each: from the host input when it is the new sequence's zeros, else one download of a few values, once per step. That download is a synchronization only on a Provider that keeps state on a Device, and then it costs next to nothing: the step starts after the previous step's logits were read for sampling, so the stream is idle. (Deriving the context from the committed token history would avoid even that, at the price of threading the history through every call that executes a graph: a prefill, a chunk, a decode and a replay. The read is kept until a Device Provider runs this model and a measurement says otherwise.) A step whose ids are not a host input (a captured replay's resident input) cannot name its rows: it runs per node.
- *Gathered form.* `ngram-embedding` gets an optional boolean attribute `gathered_rows`. When set, its third input is the rows already read, `[T * heads, head_dim]` in (token, head) order, and the operator does no lookup: it reshapes them to `[T, heads * head_dim]` and computes the next context as before. The Kernel feature `ngram-embedding-gathered-rows` is required only then. The result is byte-for-byte the dense form's, since no arithmetic touches the rows; a test checks that the rows the executor names are the rows the dense operator reads.
- *Dispatch.* At a node that reads a row table the executor reads its rows (`RowTable::read_rows`) as the node's third input and dispatches the node with `gathered_rows` set on a copy of the node; the prepared plan and the graph are those of the dense node. A Provider that does not advertise the feature rejects the node at dispatch. A node whose table is staged runs unchanged: the dense step is the default and nothing about it moves.
- *Head sizes.* A head's row count is its place in a sequence of consecutive primes above `vocab_base`, shared by every layer: layer `L`'s heads are the `L * heads + 1`-th to the `(L + 1) * heads`-th. The sequence is kept as far as it was asked for. Finding each head's prime by trial division from the base cost a layer's whole step and grew with its index (tens of milliseconds a layer at 10 million, so seconds per token over tens of layers), for the dense operator as well.
- *Loading.* The Runtime derives the row-addressed set from the architecture (`layers.{L}.ple.table` of each layer that declares an n-gram embedding) and `stream_stage_tensor` registers those instead of staging them. Because the source that maps owns its file, the Runtime keeps only the `RowTable` on the instance (no `Arc` of the source). The source verifies the table's digest while streaming it once in bounded chunks when it maps, so the integrity check keeps its guarantee without holding the table. A table counts in the residency budget as mapped bytes, never as resident weight.
- *Prefill.* A chunk reads `chunk * heads` rows per layer; sorting by offset merges adjacent pages and the chunk size bounds the gather buffer. A read that fails fails the step, with nothing partially written.

**Out of scope.** The GGUF export keeps its n-gram table in a separate file (task 5.1) and gets its own segments. CUDA does not advertise `ngram-embedding` at all yet; the gathered form needs only the host read and a small upload (a few hundred KiB per step), so it adds nothing to that gap.

**Tests.** The `ple` bundle matches `transformers` with its table row-addressed. The model instance holds no resource for the table. A step reads at most `T * heads` rows per layer and none of the rest. The dense and gathered forms agree, and a table larger than the operator's element cap runs through the gathered form.

### D7. The pinned `qwen4_exp` source and what it defines

The reference is Transformers' own `models/qwen4_exp` (Apache-2.0, "Copyright 2026 The Qwen Team and The HuggingFace Inc. team"), not Strata's engine. Transformers 5.23.0 and 5.24.1 both ship it. The executable reference is the installed 5.23.0:

| File (5.23.0) | SHA-256 |
| --- | --- |
| `modular_qwen4_exp.py` | `a5350d43d3bd8169c566200ad7efc87580d39da0371eeaf3b3e14e6f9696dde2` |
| `modeling_qwen4_exp.py` | `6bba060180c7f3c9d4d208941bbc2dfffb8933698ca332ef86179ce72355b3c0` |
| `configuration_qwen4_exp.py` | `8dfb0ef09612a7eea896672798d9a3e735b1d00e0caf935072d2cffe5afce9ab` |

5.24.1 (wheel `505b39d3e45a3715f3acb81b858eb245ce1a462373b7f99e26f0dcd41ed4cd0f`) differs only in the cache argument's name (`past_key_values` → `cache`), the model output classes and imports; its numerics are the same. The model card names no modeling file and the checkpoint's `config.json` is not read here, so tensor names and configuration values come from the loader's own tests (lot 5), not from this table.

What the source defines, as Magnetar must reproduce it:

- **Residual streams.** The embedding is repeated `hc_count` times along the feature axis, so the residual is `[T, hc_count * hidden]`. A decoder layer has no input or post-attention norm. Each block (mixer or attention, then MoE) is wrapped as:
  1. `GatedResidual(h)` gives `mixed [T, hidden]`, `h` itself, and `injection [T, hc_count]`;
  2. the block runs on `mixed`;
  3. `h' = h + block_out[:, None, :] * injection[:, :, None]` (flattened back to `[T, hc_count * hidden]`).

  `GatedResidual` is: `n = grouped_rmsnorm(h)` (groups of `hidden`, the weight `[hc_count * hidden]` as `1 + w`); `m = sigmoid(up(silu(down(n) / hc_count)))`, `[T, hc_count, hidden]`; `mixed = mean over streams of (m * n)`; `injection = 2 * sigmoid(inject(n) / hc_count)`. The final `hyper_connection_mixer` is the same without `injection`, and its output feeds the LM head directly (there is no final norm).
- **Per-layer embedding (PLE).** Only on a linear-attention layer (layer 2 in the real model). Before the layer's first `GatedResidual`, `h += PLE(h, token_ids)`: the n-gram embedding `e` (`ple_embed_dim`) feeds `key = grouped_rmsnorm(key_proj(e))` per stream and `value = value_proj(e)`; `gate = sum(key * grouped_rmsnorm(h)) / sqrt(hidden)`, then `signed_sqrt`, `sigmoid`; `gated = sigmoid(gate) * value`; the output is `gated + silu(depthwise_conv1d(grouped_rmsnorm(gated)))`, with kernel `ple_conv_kernel_size` and dilation `ngram_size`, so it keeps `(kernel - 1) * ngram_size` past columns of state.
- **N-gram ids.** Per layer and head: `ids = remainder(xor_k(shifted_token_k * multiplier_k), head_vocab_size) + head_offset`. The multipliers come from splitmix64 and the seed; head vocabulary sizes are consecutive primes above `ngram_vocab_size_base`. Tokens shift right within an EOS-delimited segment, and the two previous token ids are state, like a convolution state.
- **QSA.** An `indexed_attention` layer is the Qwen3.5 gated attention (per-head q/k norms, partial rotary, `[query | gate]` projection) plus an indexer. For each query the indexer takes the visible keys, groups complete blocks of `indexer_compress_ratio` tokens, and averages the raw index keys per block, normalizes them, and rotates them at the block's first position. The block's score is `sum over indexer heads of relu(q · k) / sqrt(index_dim)`. The `indexer_budget / ratio` best blocks are kept, every token of those blocks plus the incomplete tail block is attended, and the result masks ordinary causal attention. The indexer's raw keys `[T, index_dim]` are an extra cache per layer.
- **Dense equivalence.** With `budget = 2048` and `ratio = 4`, a query that sees at most `budget + ratio - 1 = 2051` tokens keeps every block, so QSA equals dense causal attention. Only longer contexts select.
- **MoE.** Qwen3-Next's router (softmax, top-k, `norm_topk_prob`) with a shared expert; the router normalization is checked against the existing operator in lot 3.5.
- **Linear attention.** The Qwen3.5 gated-delta mixer, with the output gate's activation `output_gate_type` (`silu` or `sigmoid`).

## Risks / Trade-offs

- **QSA and residual semantics.** They are defined by D7's pinned source, which is public and Apache-2.0. Fixtures come from running that source, never from Strata's binary engine.
- **Lot 1 replay cost.** Disabling step replay for recurrent graphs costs decode speed. Mitigation: lot 1b restores it with paired buffers.
- **Mapped experts on a slow SSD.** Misses can dominate decode. Mitigation: measure hit rate and miss latency in the same way as Strata's logs.
- **License.** The qwen-community-1.0 license must be reviewed before CI downloads the checkpoint. No weights are committed.

## Open Questions

- Ties in the indexer's top-k. PyTorch's `topk` does not define them, and the `relu` makes zero scores common (a block scores zero when every head's dot product is negative). Magnetar breaks ties toward the lower block, so a result is reproducible. The fixture generator picks a seed whose top-k has no tie at its cut, so the comparison with Transformers is exact. On the real model a tie at the cut may select a different, equally scored block than Transformers does; the qualification compares logits and tokens, not selections.
- Whether the n-gram hash and shard layout is stable across the GGUF and Hugging Face exports.
- Which GGUF tensor types ISTA-DASLab uses beyond IQ2_XS/IQ3_XXS/Q2_0, and whether they all have Magnetar kernels.
