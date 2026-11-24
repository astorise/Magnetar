## 1. Loader

- [x] 1.1 `loaders/gguf` builds the `gemma4` tokenizer; a unit test and an opt-in parity test against the upstream `tokenizer.json` (passes on the E2B file).

## 2. Runtime

- [x] 2.1 Production generation encodes plain-text prompts with the tokenizer's special tokens; chat messages and token ids unchanged (unit test).
- [x] 2.2 Untransposed block-quantized tables whose rows are whole blocks are offered packed (unit tests of the packed form and resident size).

## 3. CPU Providers

- [x] 3.1 `providers/cpu` implements and advertises the layered and latent-attention forms; the optimized Provider's attention and embedding honour them (unit test).
- [x] 3.2 The tiny Gemma 4 and DeepSeek-V2 checkpoints match `transformers` on both CPU Providers.

## 4. Real checkpoint

- [x] 4.1 E2B on the optimized CPU Provider continues "The capital of France is" with " Paris.", as an independent numpy forward of the file does.
- [x] 4.2 The tiny Gemma 4 checkpoints match `transformers` on CUDA (GPU runs 255 and 256).
- [x] 4.3 E2B on CUDA, one GPU, continues with " Paris." and the same 64 tokens as the optimized CPU Provider (GPU run 256, `gemma` group).

## 5. Validation

- [x] 5.1 `cargo fmt`, `clippy -D warnings` and the tests of `magnetar-runtime`, `providers/cpu`, `loaders/gguf` and the production-loading crate.
