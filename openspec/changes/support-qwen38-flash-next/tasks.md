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
- [ ] 2.5 CPU Provider.
  - [x] `rmsnorm.weight_offset` and a `split` kernel (with `head_count`) are implemented and advertised. The executor passes a resident `split` input through.
  - [x] The tiny hybrid checkpoint matches `transformers` on the Reference CPU Provider (1.1e-5) and on cpu-native (8.6e-6).
  - [x] cpu-native runs `gated-delta-mixer` natively.
    - Before: it fell back to the Reference kernel, which expanded the compact bf16 projection weights to f32 at every call (about 10M values per layer and token) and multiplied on one thread.
    - Now: the four input projections and the output projection are cpu-native's dense matmuls over the compact weights, across the pool. The rest runs the shared semantics, split out as `gated_delta_mixer_from_projections`, which reproduces the whole mixer bit for bit (unit test).
    - `gated_delta_mixer_reads_compact_weights` checks the outputs and both states against the Reference over a masked prefill and a decode step, with nothing expanded to f32.
    - Real 0.8B-Base: 16 tokens in 2.2 s instead of 22.6 s, still 16/16 equal to `transformers` (max |logit diff| 3.1e-5).
  - [ ] Parallel recurrence (heads across the pool) and a chunked prefill form, once long prompts are measured.
- [x] 2.6 CUDA: the same forms (norm offset, per-head split, both mixer forms), qualified on the GPU runner.
  - [x] Kernels:
    - the `rmsnorm` weight offset;
    - `split` per head, through a multi-output dispatch path;
    - `gated-delta-mixer`: projections through the existing matmul against `[out, in]` weights; convolution, gates, q/k norms, the delta rule (one thread per value column, state in registers, `key_dim` ≤ 128) and the gated norm, following the shared f32 semantics term by term.

    All three are advertised.
  - [x] A CPU emulation of the `.cu` Kernels matches `gated_delta_mixer` exactly, all three outputs, at the test geometry and at 0.8B's (16/16 heads of 128). The case is a masked five-token prefill, then a decode step from the returned states.
  - [x] GPU runner, run 45535601445 (RTX 3060, `gpu-runner-smoke.yml`'s `hybrid` group):
    - `qwen3_5_norm_offset_and_per_head_split_match_reference_cpu` and `gated_delta_mixer_matches_reference_cpu_across_steps` pass. The latter covers f32 weights and a 512-wide geometry read compact (bf16).
    - The first run (45534147734) caught `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`: the delta rule's register-held state column does not fit 1024-thread blocks. It now runs in 64-thread blocks with `__launch_bounds__`.
  - [x] The mixer's `[out, in]` projections use `gdn_project_kernel`: one warp per output column reads its contiguous weight row, f32 or bf16. The generic matmul read it with a stride of `k`. Packed weights keep the matmul path.
- [x] 2.7 Qualify Qwen3.5-0.8B-Base (revision dc7cdfe…, digest pinned) on Reference CPU, cpu-native and CUDA:
  - last-position logits against Transformers;
  - 16 greedy tokens equal;
  - evidence recorded here.

  - [x] Reference corpus: `tools/qualification/qwen3_5_reference_logits.py` runs `transformers` 5.23.0's `Qwen3_5ForConditionalGeneration` in f32, greedy, recomputing the whole sequence at each step. Prompt: "The capital of France is", 16 tokens. Output: `fixtures/qualification/qwen3.5-0.8b-base-dc7cdfe-transformers-f32.json`, bound to the weights digest `sha256:c2b1e5a1…`. The narrowest step-1 top-2 margin is 0.011.
  - [x] Reference CPU and cpu-native: `tests_qwen3_5_real_checkpoint` (ignored; `MAGNETAR_QWEN35_REAL_CHECKPOINT_DIR`, profile `f32_cross_provider`). One prefill, then 15 decode steps that carry the 18 mixers' recurrent states and the 6 attention layers' KV:

    | Provider | max \|logit diff\| | Tokens | Time (16 tokens) |
    |---|---|---|---|
    | Reference CPU | 3.2e-5 | 16/16 equal | 74.6 s |
    | cpu-native (4 threads) | 2.8e-5 | 16/16 equal | 22.6 s |
    | cpu-native, native mixer (2.5) | 3.1e-5 | 16/16 equal | 2.2 s |
    | CUDA, RTX 3060 (run 45535601445) | 1.5e-5 | 16/16 equal | 484 ms |

    Output: `[11751, 13, 561, 6511, 314, 31868, 369, 35938, 13, 561, 6511, 314, 279, 24844, 369, 36492]` (2026-11-25, x86-64 container).
  - [x] CUDA, on the GPU runner: in the table above. The same run gave cpu-native 2.9e-5, 16/16 tokens, 2.1 s on the runner's CPU.

## 3. Qwen3.8-Flash-Next blocks

- [x] 3.1 Pin a reviewable upstream modeling source for `qwen4_exp` (QSA, gated residual, n-gram PLE, MoE gate).
  - Transformers' `models/qwen4_exp` (Apache-2.0), pinned in design D7 by file SHA-256: the installed 5.23.0 is the executable reference, and 5.24.1 differs only in API names. D7 records the semantics.
  - QSA equals dense attention up to 2,051 visible tokens.
- [ ] 3.2 QSA attention with its indexer: operator, Reference semantics and fixtures.
  - [x] `qsa_select` Reference semantics (a query's kept tokens: block keys averaged, normalized, rotated at the block start, scored by `relu` sums, top `budget / ratio` blocks plus the incomplete tail). Checked against Transformers' `Qwen4ExpTextQSAIndexer` on 20 queries that prune (`the_qsa_selection_matches_the_pinned_transformers_indexer`). Ties go to the lower block, and the fixture's seed has none at the cut.
  - [x] `qsa-select` is in the catalog (queries, raw index keys and the key norm's weight in; ascending token positions padded with -1 out). The Reference CPU Providers run it, cpu-native included, and the first-native Runtime dispatches it as a graph node (`the_qsa_selection_runs_as_a_graph_node`).
  - [x] An `attention` form that reads the selection on the Reference CPU (feature `attention-token-selection`, attribute `token_selection`, the selection as the fourth input: ascending key positions per query, `-1` padding). It matches Transformers' `Qwen4ExpTextAttention` on the fixture where the selection prunes (largest difference with dense attention 2.9), and a selection naming every causal key equals dense attention (`the_selected_attention_matches_the_pinned_transformers_module`). It runs as a first-native graph node after `qsa-select` (`the_selected_attention_runs_as_a_graph_node`). Malformed selections (no readable key, fractional or unordered positions, a row missing) are refused.
  - [x] The same form on the two CPU Providers: the Reference CPU Provider advertises the feature and the optimized one hands the form to the Reference kernel, whose values the shared semantics define (`selected_attention_runs_the_reference_form_on_the_native_provider`: grouped heads, a window, a scale, a padded selection; dense attention differs where the selection skips keys). A native kernel is a speed item for long contexts and waits for a model that needs it.
  - [ ] The same form on CUDA. Until a Provider advertises the feature, the Runtime does not send it the node; a graph can attend densely instead, which equals QSA up to 2,051 visible tokens.
  - [ ] The index keys `[T, 128]` as a per-layer cache next to K and V; the pooled block keys of complete blocks never change, so a provider may cache them.
- [x] 3.3 Four-stream gated residual (`hc_count` 4, low-rank 320): operator, Reference semantics and fixtures.
  - [x] `gated_residual`, `gated_residual_inject` and `repeat_features` Reference semantics, checked against `Qwen4ExpTextGatedResidual` and the decoder layer's closing step (`the_gated_residual_matches_the_pinned_transformers_module`).
  - [x] `gated-residual` (mixed input and write weights), `gated-residual-final` (the final mixer, which has no write weights), `gated-residual-inject` and `repeat-features` are in the catalog and run on the Reference CPU Providers and as first-native graph nodes (`the_gated_residual_operators_run_as_graph_nodes`). The catalog's shape rule is `Any`; the semantics validate every shape.
- [ ] 3.4 N-gram PLE: hashed row lookup from a disk-resident table, added at layer 2; Reference semantics and fixtures.
  - [x] `ngram_embedding` (hash, EOS-delimited segments, next context) and `ngram_ple` (gate, dilated convolution, history) Reference semantics. The head primes and the hash multipliers equal Transformers' exactly; embeddings and output match on a second PLE layer, and two chunks carrying the context and history equal the whole sequence (`the_ngram_ple_matches_the_pinned_transformers_layer`).
  - [x] `ngram-embedding` (ids and context in, embeddings and next context out) and `ngram-ple` (with its convolution history as an explicit state) are in the catalog and run as first-native graph nodes, carrying both states between two runs (`the_ngram_ple_runs_as_graph_nodes_with_its_states`).
  - [ ] The table is a weight read by row and never admitted whole (4.4). Today the operator takes it as a dense input.
  - [ ] f32 tensors cannot hold the real table's row numbers (about 320M, past 2^24), so the lookup stays one operator from token ids to rows.
- [x] 3.5 MoE 512 / top-10 with a gated shared expert on the existing MoE operator.
  - The routing needs no change: Qwen3-Next's router is softmax over every expert, top-k, renormalized, with a tie to the lower expert, which `moe_route` computes with `normalize_top_k`.
  - `the_moe_block_matches_through_the_existing_operator` runs `Qwen4ExpTextSparseMoeBlock` against `moe-experts` plus a sigmoid-gated shared expert. The stacked `gate_up_proj [E, 2I, H]` and `down_proj [E, H, I]` map onto the operator's `gate`/`up [H, E*I]` and `down [I, E*H]`; the loaders do that mapping.
- [ ] 3.6 Qwen Component: recognize `qwen4_exp` and emit its graph; refuse it explicitly until 3.2–3.5 are qualified.
  - [x] The `gated-delta-mixer` operator takes `output_gate_type` (`silu`, the default, or `sigmoid`), with the kernel feature `gated-delta-sigmoid-output-gate` required only for `sigmoid`. The shared semantics, both CPU Providers (`gated_delta_graph_honours_the_output_gate_type` at the Runtime, `gated_delta_mixer_reads_compact_weights` on cpu-native) and the CUDA gated norm (`gdn_gated_norm_kernel`; its conformance test runs both gates at two geometries) implement it. `sigmoid(z) * z` equals the `silu` output element by element and the two next states do not change. The CUDA kernel compiles on the GPU runner only; the hardware run is queued with the next campaign.
  - [ ] The Component emits `output_gate_type` from the configuration (with the `model-config` 1.14.0 additions below).
  - [ ] Tiny `qwen4_exp` bundle against Transformers on the Reference CPU: layers, residual streams, the PLE layer and an indexer that prunes.

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
