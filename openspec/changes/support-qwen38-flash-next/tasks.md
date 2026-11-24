## 1. Recurrent state lifecycle (Runtime)

- [x] 1.1 Planner: `plan_execution_graph` plans recurrent graphs, with a `PreserveRecurrentState` step per state output.
- [x] 1.2 Prepared plans: the refusal is gone; state edges need no Kernel.
- [x] 1.3 Session state: `KvCache` holds each slot's committed and pending `RecurrentStateResourceBinding`, admitted under the session and released with the cache.
- [x] 1.4 Executor: inputs bind to the pending value, else the committed value, else zeros; caller-supplied state inputs are refused. Outputs are copied under attempt-unique resources once the step succeeds.
- [x] 1.5 `KvUpdateTransaction::commit` adopts the pending values after the KV layers; discarding pending KV releases them; releasing the request releases everything.
- [x] 1.6 Refusals and bypasses:
  - recurrent graphs are never captured or replayed (prefill and decode), with a traced reason;
  - their prefills are never padded (buckets and chunking turned off for the model);
  - layer segments refuse them.

  The first-native engine has no cross-request prefix reuse, continuous batching or speculative verification yet, so there is nothing for those refusals to intercept.
- [x] 1.7 Tests:
  - `recurrent_state_starts_at_zero_and_commits_only_accepted_steps`: zero start, pending versus committed, a failing step, discard, superseded and released values, restart from zero;
  - `recurrent_state_inputs_are_runtime_owned_and_need_a_cache`;
  - `gated_delta_state_lives_in_the_runtime_across_steps`: fixture prefix and pass equal the full prefill, decode after commit, retry after discard, reset;
  - planner tests in magnetar-graph and magnetar-engine.

  Engine-level checks of the guards (no padding, no capture) run with the Qwen3.5 graph in 2.7.
- [ ] 1.8 (1b) Step replay with paired committed/pending buffers, equal to the unreplayed steps.

## 2. Qwen3.5 hybrid graph (Qwen Component, Providers)

- [x] 2.1 Pin the modeling source and list the missing operators. The source is Transformers 2f1bc5e (`modeling_qwen3_5.py`, SHA-256 `c4d847b7…`); the installed 5.23.0 copy differs only in the cache argument's name.
  - Already in the catalog: per-head `rmsnorm`, the half-split `rope` over the rotary dimension (M-RoPE reduces to it for text, where all three positions are equal), grouped `attention`, the sigmoid `activation`, `mul`, the gated MLP and `gated-delta-mixer`.
  - Missing: the zero-centered norm `x · (1 + w)` and the per-head `[query | gate]` split of `q_proj`.
- [x] 2.2 Add the missing forms additively, with Reference CPU semantics and a fixture.
  - `rmsnorm.weight_offset` and `split.head_count` are gated by the `rmsnorm-weight-offset` and `split-per-head` Kernel features. The executor now forwards both attributes; it dropped them before.
  - The fixture `qwen35-attention-reference.json` comes from the pinned `Qwen3_5Attention`. `qwen35_gated_attention_block_matches_pinned_reference` matches it on the Reference CPU.
  - The CPU and CUDA Providers do not advertise these features yet (2.5, 2.6).
- [x] 2.3 Qwen Component: build the hybrid graphs.
  - [x] Runtime side: `model-config` 1.13.0 adds `linear-attention` and `decoder-variants` (record `decoder-variant-config`). Both must be read when declared, and the host keeps linking 1.2.0–1.12.0. The WIT's `recurrent-resource` comment is updated.
  - [x] Linear layers have no KV row width, and the Runtime derives their mixer weight shapes (plus the gated `q_proj` and the q/k norms).
  - [x] `gated-delta-mixer` takes an optional token mask (12 inputs), since the engine feeds no mask.
  - [x] Component 0.5.0 (digest `sha256:136e96ac…`, families `qwen3_5`/`qwen3_5_text`) reads both functions and, for Qwen3.5 only, the layer layout (a Qwen2 graph still leaves a declared layout unread, so the Runtime refuses it). It emits:
    - for a linear layer, a `gated-delta-mixer` with `conv`/`delta` recurrent slots;
    - for a full-attention layer, the gated block with per-head q/k norms and partial rotary;
    - zero-centered norms throughout.
  - [x] End to end: `tests_qwen3_5_logits::a_qwen3_5_hybrid_bundle_matches_transformers`.
    - Setup: a tiny random `qwen3_5_text` bundle, 2 linear and 2 gated full-attention layers, through the Hugging Face loader, the rebuilt Component and the Reference CPU.
    - Result: the 12 greedy tokens equal `transformers` 5.23's and the top-8 logits agree within 1.1e-5, across the prefill and 11 decode steps that carry the recurrent state.
- [x] 2.4 Hugging Face loader:
  - `qwen3_5`/`qwen3_5_text` are parsed: layer types, `linear_*` geometry, partial rotary from `rope_parameters`, output gate, and norm offset 1.
  - Mixer tensors are named without `proj`, so they keep checkpoint `[out, in]` order; `conv1d` `[C, 1, K]` becomes `[C, K]`.
  - The vision tower and MTP head are left out.
  - Tested against the pinned 0.8B-Base `config.json` and the names in its safetensors header (488 tensors).
- [ ] 2.5 CPU Provider: `rmsnorm.weight_offset` and per-head `split`. cpu-native: decode and chunked-prefill `gated-delta-mixer` kernels, equal to the Reference within tolerance including states.
- [ ] 2.6 CUDA: the same forms (norm offset, per-head split, both mixer forms), qualified on the GPU runner.
- [ ] 2.7 Qualify Qwen3.5-0.8B-Base (revision dc7cdfe…, digest pinned) on Reference CPU, cpu-native and CUDA:
  - last-position logits against Transformers;
  - 16 greedy tokens equal;
  - evidence recorded here.

## 3. Qwen3.8-Flash-Next blocks

- [ ] 3.1 Pin a reviewable upstream modeling source for `qwen4_exp` (QSA, gated residual, n-gram PLE, MoE gate), or keep this lot design-only.
- [ ] 3.2 QSA attention with its indexer: operator, Reference semantics and fixtures.
- [ ] 3.3 Four-stream gated residual (`hc_count` 4, low-rank 320): operator, Reference semantics and fixtures.
- [ ] 3.4 N-gram PLE: hashed row lookup from a disk-resident table, added at layer 2; Reference semantics and fixtures.
- [ ] 3.5 MoE 512 / top-10 with a gated shared expert on the existing MoE operator; check the routing normalization.
- [ ] 3.6 Qwen Component: recognize `qwen4_exp` and emit its graph; refuse it explicitly until 3.2–3.5 are qualified.

## 4. Expert and table residency

- [ ] 4.1 Disk-mapped and RAM expert tiers under the residency budget (task #41, `operate-budgeted-weight-residency`).
- [ ] 4.2 Adaptive GPU expert cache sized to free VRAM, with hit-rate observations.
- [ ] 4.3 CPU computation of expert misses in place (`enforce-local-multigpu-and-expert-placement`).
- [ ] 4.4 N-gram table rows read on demand, never admitted whole.

## 5. Loaders

- [ ] 5.1 GGUF: the ISTA-DASLab Qwen3.8-Flash-Next export (architecture keys, tensor types, n-gram file).
- [ ] 5.2 Hugging Face: `qwen4_exp` configuration and tensor names.

## 6. Qualification against Strata

- [ ] 6.1 Review the qwen-community-1.0 license for CI downloads.
- [ ] 6.2 Benchmark GPU suite running Magnetar on the Strata protocol (`strata_bench.py` prompts, context 8192, IQ2_XS).
- [ ] 6.3 Record decode, TTFT and 4K prefill against the reference (23.3 tokens/s, 1.15 s, 353 tokens/s), with placement, hit rate and memory.

## 7. Drafting

- [ ] 7.1 MTP drafting with per-token recurrent state snapshots for verification (`verify-speculative-and-mtp-generation`).
