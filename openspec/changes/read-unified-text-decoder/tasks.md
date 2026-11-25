## 1. Hugging Face loader

- [x] 1.1 `model.vision_embedder.` among the non-text prefixes.
- [x] 1.2 A test on the published `gemma-4-12B-it` `config.json`: family and every layer's geometry.

## 2. Validation

- [x] 2.1 `cargo fmt`, `clippy -D warnings` and tests in the Hugging Face loader; Magnetar pins it.
- [x] 2.2 The 12B GGUF metadata and `rope_freqs` (read from the file's header) give the same layout through the existing GGUF derivation.
- [x] 2.3 Real-weight logits against `transformers`, once a Component registration declares `gemma4_unified`.
  - The condition is met: the Gemma Component's manifest declares `gemma4_unified` (`openspec/specs/gemma-model-component`), and the tiny `unified` bundle (a `gemma-4-12B-it`-style config with a vision tensor to leave out) matches `transformers` at 3.5e-5 through the production path (`tests_gemma4_logits.rs`, family `gemma4_unified`).
  - What remains is the real weights: `gemma-4-12B-it` is a gated Hugging Face checkpoint (about 24 GB in bf16) and its reference logits need a `transformers` run of the 12B. Neither fits the hosted runners or this session's 9 GB of disk; it is a GPU-runner campaign with an authorized `HF_TOKEN`.
  - Campaign (authorized by the owner on 2026-11-25): the `gemma` suite of the GPU workflow downloads the public `ggml-org/gemma-4-12B-it-GGUF` `Q4_0` export (7.2 GB, pinned revision and sha256) and generates on one GPU and on the optimized CPU Provider through the Gemma Component (`real_public_gemma4_12b_q4_0_gguf_checkpoint_generates_on_cuda` and `..._on_native_cpu`). The file loads and runs on one RTX 3060: load 42 s, warm TTFT 71 ms, TPOT 31.5 ms (run 45572734242).
  - A first reading was wrong, and is recorded so it is not repeated. With the bare prompt of the other checkpoints (`The capital of France is`) the 12B continued with digits (`0111...`) on CUDA (default, without Tensor Cores, with f32 activations) and on the optimized CPU alike, which looked like a defect shared by the Providers. It is the model: inside its own chat template (`<|turn>user`, the question, `<turn|>`, `<|turn>model`, the empty thought channel) the full 48-layer model with the full vocabulary answers `The capital of France is **Paris**.` (locally on the optimized CPU Provider). The 12B tests now use that prompt.
  - Against `transformers` 5.23 in float32, on the real weights (`integration-tests/production-loading/tools/cut_real_gemma4_gguf.py` cuts the real file to a few layers and the first vocabulary rows, bytes unchanged, and dequantizes the same bytes for the reference; the Reference CPU Provider, top-8 logits over 12 greedy steps, final norm scaled by 0.01 so the soft-cap at 30 does not hide differences):

    | Cut of the real 12B | Largest logit difference |
    | --- | --- |
    | layers 0, 1 and 5 (two sliding, one full), 8,192 rows, the real prompt's ids | 8.1e-6 |
    | layers 41, 42 and 47 (a full layer first, a sliding layer, a full layer), 8,192 rows | 3.9e-6 |
    | layers 0, 1 and 5, 2,048 rows, soft-cap saturated | 1.9e-6 |

    A random bundle with the 12B's attention geometry (hidden 3840, 16 heads, 8 and 1 K/V heads, head widths 256 and 512, window 1,024, six layers) matches too: 1.0e-4 from a Hugging Face bundle and from a `gemma4` GGUF file.
  - On the GPU runner (run 45583105156, head 08e5767), the full model inside its template answers `The capital of France is **Paris**.` on both Providers: CUDA on one RTX 3060 (load 42 s, warm TTFT 265.8 ms for the 24-token prompt, TPOT 31.88 ms) and the optimized CPU Provider (3 threads, AVX2: load 36 s, TTFT 5,254 ms, TPOT 503 ms).
  - Scope: the full 12B cannot be dequantized to float32 on the hosts available (48 GB), so the logits are compared on cuts of the real weights; the full model is checked by its answer. The `gemma4_unified` Hugging Face bundle (24 GB in bf16) is not exercised on real weights: the GGUF file declares `gemma4`.
