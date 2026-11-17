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
- [ ] 3.4 A `deepseek2` GGUF and DeepSeek-V2-Lite on CUDA.

## 4. Validation

- [x] 4.1 `cargo fmt`, `clippy -D warnings` and tests of the root workspace, the Hugging Face loader and the production-loading crate.
