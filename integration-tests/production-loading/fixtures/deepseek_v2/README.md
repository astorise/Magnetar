# Tiny DeepSeek-V2 checkpoints and their `transformers` logits

Used by `src/tests_deepseek_v2_logits.rs`. Each directory holds a checkpoint
(a Hugging Face bundle or a `deepseek2` GGUF file) and `reference.json`, the
logits `transformers` 5.23's `DeepseekV2ForCausalLM` computes in float32
(`magnetar.inference-qualification.logit-reference/v1`), greedy, with the
whole sequence recomputed at every step (no KV cache).

The checkpoints are **random**, with weights scaled so that every mechanism
changes the logits. Each has 3 layers, hidden 32 and vocabulary 64. Layer 0 is
dense; layers 1 and 2 route 4 experts (top 2) beside 2 ungated shared
experts. Every layer uses latent attention: 2 heads, latent 12, 6 + 4-wide
query and key heads, 5-wide values.

| Directory | Checkpoint |
|---|---|
| `lite` | no low-rank query; YaRN (factor 4 over 16 positions) with `mscale == mscale_all_dim`, as the published V2 and V3 configurations: cosines and sines unscaled, softmax rescaled |
| `qlora` | a low-rank query (rank 8); YaRN with `mscale` 1 and `mscale_all_dim` 0.707, so cosines and sines are scaled too; routed scaling factor 2.5 |
| `plain` | no RoPE scaling |
| `gguf-kv-b` | `deepseek2` GGUF of the `wide` bundle, the latent up-projection in one `attn_kv_b` matrix (llama.cpp's converter until 2025): `Q8_0` projections and embedding, `Q4_0` experts, no `expert_gating_func` and no YaRN betas |
| `gguf-split` | the same bundle with `attn_k_b` (each head's key rows transposed) and `attn_v_b`, `head_count_kv` 1 and `key_length_mla`/`value_length_mla` (the converter since): `Q4_0` projections and embedding, `Q8_0` experts |
| `gguf-qlora` | `deepseek2` GGUF of the `wide-qlora` bundle in the split form: a low-rank query, routed scaling 2.5 (`expert_weights_scale`) |

The GGUF files come from two wider bundles, `wide` and `wide-qlora`, whose
projections' inputs are multiples of 32 so that `Q8_0` and `Q4_0` can
quantize every matrix: hidden 64, latent 96, 32 + 8-wide query and key
heads, 64-wide values, YaRN with the published `beta_fast` 32 and
`beta_slow` 1 (a GGUF file holds one YaRN multiplier, so `mscale ==
mscale_all_dim`). A GGUF file's reference is computed on its **dequantized**
weights, so the comparison tests the loader's reading of the format, not the
quantization error. The files declare no beginning-of-sequence token: the
Runtime never samples one after the first step, while a greedy reference
may.

## How they were made

With the two scripts here: Python with torch 2.15.0 (CPU), `transformers`
5.23.0 and `safetensors` 0.8.0, prompt `5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21`
and 12 greedy steps.

```bash
python make_tiny_deepseek_v2.py bundle_<v> <v>        # lite, qlora, plain, wide, wide-qlora
python deepseek_reference_logits.py bundle_<v> 12 <v>/reference.json 5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21
```

`eos_token_id` was then set to 63 in `config.json` and
`generation_config.json`, a token none of the references generates, so
generation runs the 12 steps. The weights and references do not depend on it.
Each Hugging Face directory keeps `config.json`, `model.safetensors`,
`tokenizer.json` and `tokenizer_config.json`.

The GGUF files and their references, with the `gguf` Python package
(0.19.0) as well, and the same prompt:

```bash
python make_tiny_deepseek2_gguf.py bundle_wide q_kv-b --form kv_b --proj q8_0 --experts q4_0 --embd q8_0
python make_tiny_deepseek2_gguf.py bundle_wide q_split --form split --proj q4_0 --experts q8_0 --embd q4_0
python make_tiny_deepseek2_gguf.py bundle_wide-qlora q_qlora --form split --proj q8_0 --experts q4_0 --embd q8_0
python deepseek_reference_logits.py q_<v>/ref 12 gguf-<v>/reference.json 5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21
```

Each GGUF directory keeps `model.gguf` (`q_<v>/gguf/model.gguf`).
