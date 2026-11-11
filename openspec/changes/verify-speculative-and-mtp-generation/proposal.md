## Why

Single-stream decode on Magnetar's CUDA profile is bandwidth-bound: every token reads all active weights once (Mistral 7B `Q8_0`: 28 ms per token on an RTX 3060, about 270-305 GB/s on its projections). The decode matmul Kernels already serve up to eight rows of activations per weight read (`Q8_MATMUL_ROWS`), so checking several candidate tokens in one target pass costs about as much as producing one. Speculative decoding turns that into throughput: propose `k` tokens cheaply, verify them in one target pass, keep the longest accepted prefix plus the target's own next token.

Public measurements confirm the size of the effect and its limits. A November 2026 test of multi-token-prediction (MTP) heads with llama.cpp (Qwen3.8 Flash-Next, 125B MoE, NVIDIA RTX Spark and Apple M5 Max) reports +62 % on Python code (31.5 to 50.9 tokens/s) and +72 % (47.9 to 82.6) with more than 80 % of drafts accepted, +33-36 % on French prose with 58-59 % accepted, and no effect on prefill.

The roadmap change this proposal belonged to stayed generic (E08 exact speculative decoding, E09 adaptive MTP verification, F03 draft/MTP model semantics). This revision makes it concrete and staged so each stage ships and is qualified alone:

1. **n-gram (prompt lookup) drafts**: no extra model, best on code and on answers that copy their context.
2. **A small draft model** of the same tokenizer family (for example Qwen2.5-0.5B drafting for a Qwen2.5 7B target).
3. **MTP heads** once a Model Component declares them (DeepSeek-V3, Qwen3-Next and similar families; none of today's Components do).

## What Changes

- **Runtime verification (all stages)**: a verify step runs the target on the last committed token plus `k` candidates in one pass, accepts the longest prefix the target agrees with, appends the target's correction, and commits KV rows only for accepted tokens. Greedy output is token-for-token identical to target-only decoding; stochastic sampling uses rejection sampling that preserves the target distribution.
- **Graph contract (additive WIT, `model-component-graph@1.6.0`)**: `build-extend-graph(cached-token-count, sequence-length)` builds a graph for `sequence-length` new tokens on top of `cached-token-count` committed ones (position offset, KV append, causal attention over history plus the new tokens). Components already build such graphs internally; only the export is new. Components that do not export it keep working; speculation is then Unsupported for them.
- **Stage 1**: an n-gram draft source in the Runtime (suffix match over the prompt and generated tokens).
- **Stage 2**: a draft-model source: a second Model Instance sharing the target's tokenizer, with its own KV, resynchronized to the accepted tokens after each verification.
- **Stage 3**: MTP: an additive, versioned graph-contract extension through which a Component declares MTP heads and emits their candidate logits; the Runtime treats them as another draft source with the same verification.
- **Controls**: speculation is off by default, opt-in per Model Instance or request, with a bounded draft length and adaptive length within that bound; a disable control and per-request statistics (proposed, accepted, acceptance rate).

## Capabilities

### New Capabilities

- `speculative-generation`: draft sources, exact verification, KV commit rules, controls and statistics.

### Modified Capabilities

- `model-component-graph-contract`: additive `build-extend-graph` (stage 1) and, later, an additive MTP head declaration (stage 3).

## Impact

- **Owning layers**: Runtime (draft sources, verification, acceptance, KV commit, scheduling); Model Components (exporting extend graphs, later MTP heads); Providers (unchanged contracts: multi-row decode Kernels and attention over history plus new tokens already exist in `providers/cuda` and `providers/cpu`; any gap is a Provider task, not a contract change).
- **WIT**: one additive function in a new minor version; existing exports and signatures unchanged; old Components link and run without speculation.
- **Resource Affinity**: draft and target state stay bound to their own Model Instance, Provider and Device; a draft model may run on another Device of the same Provider or another Provider only through normal resolution. KV and drafts are request-scoped and never shared across sessions.
- **Authority**: none added. Draft sources read only the request's own tokens; a draft model is an ordinary, trust-verified Model Instance.
- **Memory**: verification adds up to `k` rows of activations and KV per step; a draft model adds its weights and KV and goes through Memory Manager admission like any instance.

## Dependencies

- `A08` [`qualify-real-checkpoint-performance`](../qualify-real-checkpoint-performance/proposal.md) for the campaigns.
- `B10` [`keep-sampling-on-device`](../keep-sampling-on-device/proposal.md) for stochastic acceptance on the Device (greedy acceptance does not need it).
- `E01` [`serve-continuous-batches-with-qos`](../serve-continuous-batches-with-qos/proposal.md) only for speculation inside continuous batches.
- `F02` [`execute-and-cache-moe-experts`](../execute-and-cache-moe-experts/proposal.md) only for MoE MTP profiles.

## Non-goals

- No speculation during prefill (the measured gain is decode-only).
- No change to sampling semantics: speculation never changes greedy output and never changes the stochastic output distribution.
- No Component control over acceptance, commits, Providers or Devices.
- No tree-structured drafts (multiple branches per step) in this change.
- No automatic draft-model download or selection outside explicit configuration.
- No claim of a speedup before the qualification campaigns below.
