## Context

Today a decode step runs the Component's decode graph for one token (`build-decode-graph(cached-token-count)`), then commits its KV rows. With in-place KV history (`append-kv-history-in-place`), a step writes its rows beyond the committed count, and commit only advances that count. A captured decode step replays with the position as an advancing attribute (`capture-and-replay-cuda-graphs`).

The CUDA decode Kernels already take several rows per weight read: `q8_matmul_rows_kernel`, `q4k_matmul_rows_kernel` and their gated forms serve up to `Q8_MATMUL_ROWS` = 8 rows per block and stay on the packed path up to 64 rows. Verifying `k` drafts therefore reads the weights once, like a single decode step, as long as `k + 1 <= 8`.

Components already build graphs for several new tokens over a history: `components/qwen` builds every graph through `build_graph(phase, sequence_length, position_offset, kv_append)`; the WIT world only exports the one-token decode case.

## Goals / Non-Goals

**Goals:** exact (greedy-identical, distribution-preserving) speculative decoding; three draft sources in order (n-gram, draft model, MTP); measured gains on real checkpoints; a disable control; no new authority.

**Non-Goals:** see proposal (no prefill speculation, no tree drafts, no sampling change, no Component-owned acceptance).

## Decisions

### 1. One verification step, any draft source

```
committed: t0 .. tn          drafts d1..dk (from a draft source)
verify graph: tokens [tn, d1, ..., dk] at positions n .. n+k   (extend graph, k+1 rows)
target selections:  s1 .. s(k+1)   (s_i = target's choice after position n+i-1)
accept d1..dj while d_i == s_i; generated = d1..dj, s(j+1)
commit KV rows n .. n+j (tn's row was already committed; rows for d1..dj)
```

The extend graph writes KV for all `k + 1` positions in place; committing `j + 1` advances the committed count past `tn` and the accepted drafts only, so rows of rejected drafts are overwritten by the next step, never read (attention reads committed plus current rows). No copy or rollback is needed.

Greedy acceptance compares token ids. Stochastic acceptance uses the standard rejection rule (accept `d_i` with probability `min(1, p_target(d_i) / p_draft(d_i))`, otherwise sample from the normalized `max(0, p_target - p_draft)`); an n-gram draft is treated as a deterministic draft (`p_draft = 1` on its token). Stochastic acceptance needs the target's distribution per row on the Device; until `keep-sampling-on-device` provides it, stochastic requests run target-only.

### 2. Graph contract: `build-extend-graph`

`model-component-graph@1.6.0` adds:

```wit
/// A graph for `sequence-length` new tokens appended to
/// `cached-token-count` committed ones: positions start at
/// `cached-token-count`, KV is appended in place, attention is causal
/// over the committed history and the new tokens.
export build-extend-graph: func(cached-token-count: u64, sequence-length: u64) -> string;
```

`build-decode-graph(c)` equals `build-extend-graph(c, 1)` for Components that export both. The Runtime checks the export at link time; absent, speculation is Unsupported for that instance. `components/qwen` and `components/llama` export it from their existing `build_graph`. Operator requirement: `attention` with `T` queries over `cached + T` keys, causal aligned to the last key; Reference CPU, `providers/cpu` and `providers/cuda` are checked (and extended if needed) against the reference before stage 1 lands.

### 3. Draft sources

- **n-gram (stage 1)**: find the longest suffix (up to `n` tokens, default 3) of the request's tokens that also occurs earlier in the prompt or output; propose the `k` tokens that followed its most recent earlier occurrence. Host-side, O(length) per step with a rolling index; no model.
- **Draft model (stage 2)**: a second Model Instance, loaded and trust-verified like any other, with the same tokenizer (checked by tokenizer identity). It decodes `k` tokens with its own KV, then, after verification, its KV is advanced or trimmed to the accepted prefix (same in-place rule) and fed the target's correction. It may live on another Device; resolution and affinity apply as for any instance.
- **MTP (stage 3)**: a Component declares heads (count, output edges); the decode graph also emits their logits; the Runtime takes their argmax (greedy) as the next step's drafts. Needs a Component for a model family that ships MTP heads, which is part of the DeepSeek/Qwen3-Next Component work.

### 4. Capture and scheduling

A verify step with a fixed `k` has fixed shapes, so it is captured and replayed like a decode step (position and key count advancing). The draft length is adapted only among a few captured lengths (for example 1, 2, 4), each captured once. If acceptance makes speculation slower than target-only decode (measured per request over a window), the request falls back to target-only steps.

### 5. Controls and reporting

`GenerationParameters` gains an optional speculation setting (source, maximum `k`); a Model Instance setting gives the default; an environment variable mirrors the disable control for CI. Usage reports drafted and accepted counts; campaigns report acceptance rate per prompt class, TPOT, and tokens per verification step.

## Risks / Trade-offs

- [Gains depend on acceptance: about 80 % on code, about 58 % on prose in the public measurement] -> report per prompt class; adaptive length and automatic fallback bound the loss.
- [Verification costs more than one decode when `k + 1` exceeds the Kernels' row tile, or for attention over long histories] -> cap `k` at 7 by default; measure verify-step time against decode-step time on each profile.
- [Numerical differences between the one-row and multi-row Kernel paths could change greedy ties] -> the greedy-identity tests compare against target-only runs on the same Provider; a mismatch blocks the profile.
- [A draft model costs memory and its own decode time] -> stage 2 is opt-in and qualified on pairs where the draft is at least 10x smaller.

## Migration Plan

Stage 1 (extend graph, verification, n-gram) lands behind the disabled-by-default control; stages 2 and 3 follow independently. Rollback is disabling speculation; no persisted state changes.

## Open Questions

- Default `k` and n-gram length per profile, from the first campaign.
- Whether the draft model shares the target's Provider stream (simplest) or runs concurrently on another Device.
- The exact MTP head declaration, once a Component for an MTP model family exists.
