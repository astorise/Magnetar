# Tiny Gemma 4 checkpoints and their `transformers` logits

Used by `src/tests_gemma4_logits.rs`. Each directory holds a checkpoint and
`reference.json`, the logits `transformers`' `Gemma4ForCausalLM` computes in
float32 (`magnetar.inference-qualification.logit-reference/v1`).

The checkpoints are **random**: 6 layers, hidden 64, vocabulary 64, 8
experts. Their weights are scaled so that every mechanism changes the
logits: a 6-token sliding window, norm gains and `layer_scalar` away from 1,
logits large enough for the soft-cap. They test the Runtime's handling of
the Gemma 4 decoder, not a real model.

| Directory | Checkpoint |
|---|---|
| `dense` | Hugging Face bundle (nested `text_config`, `model.language_model.*`): sliding and full layers, V from K, partial RoPE, soft-cap |
| `unified` | the same decoder as a `gemma4_unified` bundle, with a `model.vision_embedder.*` tensor to leave out |
| `gguf-dense` | `gemma4` GGUF: `Q4_0` projections, `Q8_0` tied embedding, layout from the metadata |
| `gguf-shared` | plus shared KV on the last layers and a wider MLP |
| `gguf-ple` | plus per-layer input tables (`Q8_0`) |
| `gguf-moe` | plus a dense MLP beside 8 routed experts (fused `Q4_0` gate and up, router scale, per-expert scale) |

A GGUF file's reference is computed on its **dequantized** weights, so the
comparison tests the format's handling, not the quantization error.

## How they were made

With the scripts of `components/gemma` at `9186a0c`, under
`tools/qualification/`. The environment was Python with torch 2.15.0 (CPU),
`transformers` 5.23.0, numpy 2.5.5, `safetensors` 0.8.0 and `gguf`. The
prompt was `5 17 33 8 41 2 60 12`, with 12 greedy steps:

```bash
python make_tiny_gemma4_bundle.py bundle_<v> --variant <v>   # dense, unified, shared, ple, moe
python make_tiny_gemma4_gguf.py bundle_<v> q4_<v>            # GGUF in q4_<v>/gguf, dequantized reference bundle in q4_<v>/ref
python gemma4_reference_logits.py bundle_<v> 12 dense/reference.json 5 17 33 8 41 2 60 12
python gemma4_reference_logits.py q4_<v>/ref 12 gguf-<v>/reference.json 5 17 33 8 41 2 60 12
```

The Hugging Face directories keep `config.json`, `model.safetensors`,
`tokenizer.json` and `tokenizer_config.json`; the GGUF directories keep
`model.gguf`.
