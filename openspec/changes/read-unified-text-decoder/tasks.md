## 1. Hugging Face loader

- [x] 1.1 `model.vision_embedder.` among the non-text prefixes.
- [x] 1.2 A test on the published `gemma-4-12B-it` `config.json`: family and every layer's geometry.

## 2. Validation

- [x] 2.1 `cargo fmt`, `clippy -D warnings` and tests in the Hugging Face loader; Magnetar pins it.
- [x] 2.2 The 12B GGUF metadata and `rope_freqs` (read from the file's header) give the same layout through the existing GGUF derivation.
- [ ] 2.3 Real-weight logits against `transformers`, once a Component registration declares `gemma4_unified`.
  - The condition is met: the Gemma Component's manifest declares `gemma4_unified` (`openspec/specs/gemma-model-component`), and the tiny `unified` bundle (a `gemma-4-12B-it`-style config with a vision tensor to leave out) matches `transformers` at 3.5e-5 through the production path (`tests_gemma4_logits.rs`, family `gemma4_unified`).
  - What remains is the real weights: `gemma-4-12B-it` is a gated Hugging Face checkpoint (about 24 GB in bf16) and its reference logits need a `transformers` run of the 12B. Neither fits the hosted runners or this session's 9 GB of disk; it is a GPU-runner campaign with an authorized `HF_TOKEN`.
