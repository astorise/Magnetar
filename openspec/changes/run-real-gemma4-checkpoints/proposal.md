## Why

Issue #187 left one item open: a real Gemma 4 checkpoint. The Gemma 4
Component (`register-gemma4-model-component`) was only checked against
tiny random checkpoints. Running Google's Gemma 4 E2B instruct QAT `Q4_0`
GGUF file (`google/gemma-4-E2B-it-qat-q4_0-gguf`, revision `675cff4`)
showed four gaps:

- the GGUF loader refused its `gemma4` tokenizer;
- `providers/cpu` implemented none of the layered or latent-attention
  operator forms, so neither CPU Provider could run Gemma 4 or DeepSeek;
- a production generation encoded a plain-text prompt without the
  tokenizer's special tokens, so the E2B prompt had no
  beginning-of-sequence token. Without it the model repeats the prompt
  (" France is France is …"); with it, it answers " Paris.". An
  independent numpy forward of the same file, transcribing llama.cpp's
  Gemma 4 graph, gives the same two continuations;
- weight loading dequantized every untransposed k-quant table to `f32`.
  The E2B file's `Q6_K` token table and its 35 per-layer input tables
  (262,144 rows each) then take about 11 GB, and loading ran out of memory
  on a 12 GB GPU.

## What Changes

- **GGUF tokenizer** (`loaders/gguf`): `gemma4` builds a byte-fallback BPE
  from the file's pieces and merges, spaces as `▁`, no added leading space.
  Its encoding and decoding match the upstream `tokenizer.json` on the E2B
  file (opt-in parity test).
- **Plain-text prompts** (`magnetar-runtime`, production generation): encoded
  with the tokenizer's special tokens, as `transformers`' `tokenizer(text)`
  and llama.cpp do. Chat messages, whose template writes those tokens, and
  token ids are unchanged.
- **Quantized tables stay packed** (`magnetar-runtime`, weight loading): an
  untransposed block-quantized matrix whose rows are whole blocks is offered
  packed, as projections are. The embedding Kernels of the CUDA and the
  optimized CPU Providers already read packed tables; a Provider that
  cannot declines it and receives `f32`.
- **CPU Providers** (`providers/cpu`): the Reference CPU and optimized CPU
  Providers implement and advertise the forms the Runtime's own Reference
  CPU implements. The optimized Provider's attention and embedding honour
  `scale`, and leave another V width to the Reference CPU kernel.
- **Tests** (`integration-tests/production-loading`):
  - the tiny Gemma 4 and DeepSeek-V2 checkpoints on both CPU Providers and,
    for Gemma 4, on CUDA;
  - the real E2B file on CUDA and on the optimized CPU Provider, pinned by
    revision and sha256.
- **GPU workflow**: a `gemma` group runs the tiny checkpoints and E2B on
  CUDA. The continuation of "The capital of France is" must start with
  "Paris".

## Evidence

Tiny checkpoints against `transformers` (max |logit difference|, identical
tokens):

| Checkpoint | Reference CPU Provider | Optimized CPU (f32 activations) |
|---|---|---|
| Gemma 4 `dense` / `unified` | 3.481e-5 / 3.481e-5 | 3.433e-5 / 3.433e-5 |
| Gemma 4 `gguf-dense` / `gguf-shared` | 3.338e-5 / 1.144e-4 | 2.575e-5 / 1.030e-4 |
| Gemma 4 `gguf-ple` / `gguf-moe` | 2.718e-5 / 3.004e-5 | 2.432e-5 / 2.527e-5 |
| DeepSeek `lite` / `qlora` / `plain` | 6.676e-6 / 1.240e-5 / 3.338e-6 | 6.676e-6 / 1.240e-5 / 3.338e-6 |
| DeepSeek `gguf-kv-b` / `gguf-split` / `gguf-qlora` | 7.629e-6 / 9.537e-6 / 1.097e-5 | 7.629e-6 / 9.537e-6 / 1.097e-5 |

On CUDA with f32 activations (GPU run 256), the tiny Gemma 4 checkpoints
give: `dense` 3.147e-5, `unified` 3.147e-5, `gguf-dense` 4.387e-5,
`gguf-shared` 5.198e-5, `gguf-ple` 3.719e-5, `gguf-moe` 4.101e-5.

Gemma 4 E2B QAT `Q4_0`, prompt "The capital of France is", greedy, packed
weights, int8 activations:

| Provider | Load | Warm TTFT | TPOT (64 tokens) |
|---|---|---|---|
| CUDA, one RTX 3060 (GPU run 256) | 33.0 s | 20.2 ms | 11.22 ms |
| Optimized CPU, the runner's 3 threads, AVX2 (GPU run 256) | 31.8 s | 555 ms | 186.5 ms |
| Optimized CPU, 4 threads, AVX2 (local) | 40.1 s | 624 ms | 250.1 ms |

All three generate " Paris." and the same 64 token ids. Their first three
tokens (9079 " Paris", 236761 ".", 106 `<turn|>`) are those of the numpy
forward. The test does not stop at the end of the turn, so the tokens after
106 carry no meaning.

**Limits.** One prompt and greedy decoding; no logit comparison of the
real file with `transformers`. The numpy forward transcribes llama.cpp's
graph; it is not llama.cpp itself.

## Impact

- `loaders/gguf`: `src/tokenizer.rs`.
- `providers/cpu`: `src/lib.rs`, `src/native/`.
- `magnetar-runtime`: `first_native_runtime/production.rs`,
  `model_loading.rs` and their tests.
- `integration-tests/production-loading`: `logit_comparison.rs`,
  `tests_gemma4_logits.rs`, `tests_deepseek_v2_logits.rs`,
  `tests_gguf_real_checkpoint_smoke.rs`.
- `.github/workflows/gpu-runner-smoke.yml`.
- A plain-text prompt to a model whose tokenizer adds a
  beginning-of-sequence token (Llama, Mistral, Gemma, DeepSeek) now starts
  with it, so those models' continuations change.
