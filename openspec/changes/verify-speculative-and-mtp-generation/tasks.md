## 1. Contract and operators

- [ ] 1.1 Add `build-extend-graph(cached-token-count, sequence-length)` to `model-component-graph@1.6.0` (additive); validate old/new linking; export it from `components/qwen` and `components/llama`; rebuild and repin their fixtures.
- [ ] 1.2 Check `attention` with `T` queries over `cached + T` keys (causal aligned to the last key) in Reference CPU, `providers/cpu` and `providers/cuda` against the reference; extend any Provider that lacks it.
- [ ] 1.3 Report speculation Unsupported for instances whose Component lacks the export.

## 2. Verification and n-gram drafts (stage 1)

- [ ] 2.1 Implement the verify step: extend graph over the last committed token plus `k` drafts, greedy acceptance, the target's correction, KV commit of accepted rows only.
- [ ] 2.2 Implement the n-gram draft source over the request's prompt and generated tokens.
- [ ] 2.3 Add `GenerationParameters` speculation settings, the instance default, the disable control and drafted/accepted usage counts.
- [ ] 2.4 Capture and replay verify steps for a small set of draft lengths; adaptive length within the bound; per-request fallback to target-only steps.
- [ ] 2.5 Tests: greedy identity with target-only generation (tiny bundle on Reference CPU, real checkpoint on CUDA); all-rejected and all-accepted steps; cancellation during verification leaves committed KV unchanged; no cross-request draft input.
- [ ] 2.6 Campaign: Qwen2.5 and Mistral 7B (`Q8_0`, `Q4_K_M`) on the RTX 3060, code and prose prompts: acceptance rate, tokens per step, TPOT with and without speculation.

## 3. Draft model (stage 2)

- [ ] 3.1 Implement the draft-model source: a second trust-verified Model Instance with tokenizer identity checked, its KV advanced or trimmed to the accepted prefix.
- [ ] 3.2 Implement stochastic acceptance (rejection sampling) once on-device sampling provides per-row distributions; until then stochastic requests stay target-only.
- [ ] 3.3 Tests: tokenizer mismatch rejected; greedy identity; a statistical test of the output distribution under stochastic sampling.
- [ ] 3.4 Campaign: Qwen2.5-0.5B drafting for a Qwen2.5 7B target (or the nearest available pair), including memory and the draft's own cost.

## 4. MTP heads (stage 3)

- [ ] 4.1 Add the additive MTP head declaration to the graph contract.
- [ ] 4.2 Read declared heads' candidates as a draft source; Unsupported for Components without heads.
- [ ] 4.3 Qualify on the first Component for a model family that ships MTP heads (DeepSeek-V3 / Qwen3-Next class), with the same identity tests and campaign.

## 5. Validation and documentation

- [ ] 5.1 cargo fmt/clippy/tests in the Runtime, affected Providers and Components; WIT validation; native and `wasm32` builds.
- [ ] 5.2 Document the controls, the exactness guarantee, the measured gains per prompt class and the limits (no prefill effect, acceptance-dependent).
- [ ] 5.3 Update the roadmap coverage index (E08 for stages 1-2, F03 and E09 for stage 3) only with qualified evidence.
