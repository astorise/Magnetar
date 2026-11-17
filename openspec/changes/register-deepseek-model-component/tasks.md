## 1. Runtime

- [x] 1.1 Remove the hidden-size rule from `ModelComponentArchitectureMetadata::validate`; the test that expected the refusal now expects acceptance, and a zero head dimension is still refused.
- [x] 1.2 Hugging Face loader: refuse greedy or group-limited routing with `norm_topk_prob = true`; tested on the V2-Lite configuration, and V3's `noaux_tc` still renormalizes.

## 2. The Component

- [x] 2.1 Pin `components/deepseek` at `a82132f`; describe it in `SUBMODULES.md`.
- [x] 2.2 Build it with the pinned toolchain; check in `deepseek-real.component.wasm` and its manifest (WIT `1.7.0`, families `deepseek`, `deepseek_v2`, `deepseek_v3`, `deepseek2`, formats `huggingface` and `gguf`).

## 3. Acceptance

- [x] 3.1 Three tiny DeepSeek-V2 bundles and their `transformers` references, with the scripts that made them.
- [x] 3.2 `tests_deepseek_v2_logits.rs` through the shared `logit_comparison.rs`; `deepseek-real` among the real Components.
- [x] 3.3 A negative control: the `lite` bundle against a reference without YaRN fails.
- [x] 3.4 Three tiny `deepseek2` GGUF files (both forms of the latent up-projection, `Q8_0` and `Q4_0`) with references on their dequantized weights, compared on the Reference CPU; negative controls without YaRN and without the routed scaling fail.
- [x] 3.5 GGUF loader: a byte-level BPE adds no space before the text; parity with DeepSeek-V2-Lite-Chat's `tokenizer.json` on its GGUF header.
- [x] 3.5a CUDA Provider: the rope dispatch reads YaRN before the banded scaling (found by the first GPU run); the Qwen1.5-MoE steps require "Paris" in the continuation, which the corrected tokenization turned into a multiple-choice question.
- [ ] 3.6 The six tiny checkpoints on CUDA, f32 activations (GPU workflow, `moe` suite).
- [ ] 3.7 DeepSeek-V2-Lite-Chat `Q8_0` GGUF across two GPUs continues "The capital of France is" with "Paris" (GPU workflow, `moe` suite).

## 4. Validation

- [x] 4.1 `cargo fmt`, `clippy -D warnings` and tests of the root workspace, the Hugging Face loader and the production-loading crate.
- [x] 4.2 `cargo fmt`, `clippy -D warnings` and tests of the GGUF loader; the production-loading crate again.
