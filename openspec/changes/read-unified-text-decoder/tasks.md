## 1. Hugging Face loader

- [x] 1.1 `model.vision_embedder.` among the non-text prefixes.
- [x] 1.2 A test on the published `gemma-4-12B-it` `config.json`: family and every layer's geometry.

## 2. Validation

- [x] 2.1 `cargo fmt`, `clippy -D warnings` and tests in the Hugging Face loader; Magnetar pins it.
- [x] 2.2 The 12B GGUF metadata and `rope_freqs` (read from the file's header) give the same layout through the existing GGUF derivation.
- [ ] 2.3 Real-weight logits against `transformers`, once a Component registration declares `gemma4_unified`.
