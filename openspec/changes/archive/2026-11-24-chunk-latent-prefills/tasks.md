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

- [x] 5.1 The CUDA conformance test and the tiny checkpoints on CUDA pass (run 44404969859, largest logit difference 9.8e-6).
- [x] 5.2 DeepSeek-V2-Lite-Chat Q8_0 across both GPUs prefills in passes of 2,048 rows: 16,384 tokens with an `f32` history and 32,768 tokens; TTFT and TPOT recorded in `cache-latent-attention-history` task 6.7 (run 44404969859, design.md Evidence).
- [x] 5.3 The `tq4` prefill's extra time over `f32`'s. Not the latent attention pass (`latent_prefill_attention_times`, run 44417598937: equal times); the DeepSeek phases step (run 44431584039) spreads it over every operator, those that never read the history included, its KV append adding 5 and 37 ms at 1,024 and 16,384 tokens (design.md, Evidence).
- [x] 5.4 The GPUs' clocks over the DeepSeek long-context step, to tell whether the later requests run on slower clocks: the second GPU's do (run 44441670479: 94 C, SM 742-1,320 MHz from the end of the `f32` requests on; the first GPU at most 64 C).
- [x] 5.5 Whether a model loaded after another runs every request slower whatever its history: it does (run 44450645042: `f32` loaded third 8% and 13% slower than loaded first at 1,024 and 4,096 tokens, `tq4` second 3% and 4%).
- [x] 5.6 Freed device memory kept in the pool, each length run twice within a model (run 44458793626): requests slow down within a model too, every heavy Kernel alike, as the second GPU throttles (91-95 C, 675-1,320 MHz); the pool change reverted (providers/cuda 36c3717).
