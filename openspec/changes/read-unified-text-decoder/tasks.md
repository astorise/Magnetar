## 1. Hugging Face loader

- [x] 1.1 `model.vision_embedder.` among the non-text prefixes.
- [x] 1.2 A test on the published `gemma-4-12B-it` `config.json`: family and every layer's geometry.

## 2. Validation

- [x] 2.1 `cargo fmt`, `clippy -D warnings` and tests in the Hugging Face loader; Magnetar pins it.
- [x] 2.2 The 12B GGUF metadata and `rope_freqs` (read from the file's header) give the same layout through the existing GGUF derivation.
- [ ] 2.3 Real-weight logits against `transformers`, once a Component registration declares `gemma4_unified`.
  - The condition is met: the Gemma Component's manifest declares `gemma4_unified` (`openspec/specs/gemma-model-component`), and the tiny `unified` bundle (a `gemma-4-12B-it`-style config with a vision tensor to leave out) matches `transformers` at 3.5e-5 through the production path (`tests_gemma4_logits.rs`, family `gemma4_unified`).
  - What remains is the real weights: `gemma-4-12B-it` is a gated Hugging Face checkpoint (about 24 GB in bf16) and its reference logits need a `transformers` run of the 12B. Neither fits the hosted runners or this session's 9 GB of disk; it is a GPU-runner campaign with an authorized `HF_TOKEN`.
  - Campaign (authorized by the owner on 2026-11-25): the `gemma` suite of the GPU workflow downloads the public `ggml-org/gemma-4-12B-it-GGUF` `Q4_0` export (7.2 GB, pinned revision and sha256) and generates on one GPU through the Gemma Component (`real_public_gemma4_12b_q4_0_gguf_checkpoint_generates_on_cuda`), which exercises the 12B's real layout. The GGUF declares `gemma4`, so the `gemma4_unified` Hugging Face path and a `transformers` reference stay out of reach: the bf16 bundle is 24 GB, as large as the runner's two 12 GB GPUs together.
