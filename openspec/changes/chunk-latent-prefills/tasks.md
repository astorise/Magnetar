## 1. Contract

- [x] 1.1 `magnetar-core`: `latent_kv_width` (`attention` attribute), `ATTENTION_LATENT_KV_FEATURE` required by it or by more than three inputs; `attention` takes 3 to 5 inputs.
- [x] 1.2 `magnetar-provider-api`: `operator_semantics::latent_keys` and `latent_values`.

## 2. Runtime and Providers

- [x] 2.1 Dispatch of a five-input `attention` (`dispatch_attention_with_latent_weights`), `latent_kv_width` forwarded; the Reference CPU expands and advertises the feature.
- [x] 2.2 `providers/cpu`: the reference path expands, the native path declines the form; advertised.
- [x] 2.3 `providers/cuda`: `attention_latent_kv` (`latent_split_kernel`, the matmul, the head concatenation, the tiled attention; a TurboQuant history dequantized), the projections kept compact; advertised; `attention_over_latent_keys_and_values_matches_the_cpu`.

## 3. Component

- [x] 3.1 `components/deepseek`: the prefill attends through the form over the history (module doc updated); the fixture Component rebuilt reproducibly and its digest repinned (`sha256:37f4e9e5...`).

## 4. Tests

- [x] 4.1 `a_latent_attention_prefill_runs_in_passes` (was `..._runs_in_one_pass`): every tiny checkpoint keeps passes of 4 rows and matches `transformers`; the other DeepSeek tests (CPU Providers, TurboQuant histories, exact head) pass; `magnetar-core`, `-graph`, `-provider-api`, `-engine`, `-runtime`, the CPU Provider and the production-loading suite pass.

## 5. GPU qualification

- [ ] 5.1 The CUDA conformance test and the tiny checkpoints on CUDA pass.
- [ ] 5.2 DeepSeek-V2-Lite-Chat Q8_0 across both GPUs prefills in passes of 2,048 rows: 16,384 tokens with an `f32` history and 32,768 tokens; TTFT and TPOT recorded in `cache-latent-attention-history` task 6.7.
