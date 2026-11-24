## 1. Recurrent state lifecycle (Runtime)

- [ ] 1.1 Planner: drop the recurrent refusal in `plan_execution_graph`; keep recurrent output edges live to the end of the step.
- [ ] 1.2 Prepared plans: drop the refusal; record each plan's recurrent bindings.
- [ ] 1.3 Session state: map each binding to a committed resource, admitted at its fixed size and zero-initialized before the first commit.
- [ ] 1.4 Executor: bind inputs to committed resources and write outputs to attempt-unique pending resources.
- [ ] 1.5 `KvUpdateTransaction`: promote, commit and abort recurrent slots together with KV layers; release them with the session.
- [ ] 1.6 Refusals and bypasses: prefix reuse bypassed with an observation; step capture skipped with a traced reason; batching and speculative verification refused.
- [ ] 1.7 Tests on synthetic graphs:
  - accumulation across steps;
  - zero start for a new sequence;
  - prefill → decode continuity;
  - chunked prefill equal to one-pass prefill;
  - a failed step that leaves state and KV unchanged;
  - a `gated-delta-mixer` graph equal to the fixture's prefix and continuation.
- [ ] 1.8 (1b) Step replay with paired committed/pending buffers, equal to the unreplayed steps.

## 2. Qwen3.5 hybrid graph (Qwen Component, Providers)

- [ ] 2.1 Pin the Transformers Qwen3.5 modeling source used for semantics and fixtures; list its attention-block operators missing from the catalog.
- [ ] 2.2 Add the missing operators additively with Reference CPU semantics and fixtures.
- [ ] 2.3 Qwen Component: recognize the Qwen3.5 text configuration; emit the hybrid prefill and decode graphs with `conv`/`delta` recurrent slots; rebuild and repin the wasm.
- [ ] 2.4 Hugging Face loader: map the Qwen3.5 tensor names and configuration.
- [ ] 2.5 cpu-native: decode and chunked-prefill `gated-delta-mixer` kernels, equal to the Reference within tolerance including states.
- [ ] 2.6 CUDA: the same two forms, qualified on the GPU runner.
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
