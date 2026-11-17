# Tiny DeepSeek-V2 checkpoints and their `transformers` logits

Used by `src/tests_deepseek_v2_logits.rs`. Each directory holds a Hugging Face
bundle and `reference.json`, the logits `transformers` 5.23's
`DeepseekV2ForCausalLM` computes in float32
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

## How they were made

With the two scripts here: Python with torch 2.15.0 (CPU), `transformers`
5.23.0 and `safetensors` 0.8.0, prompt `5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21`
and 12 greedy steps.

```bash
python make_tiny_deepseek_v2.py bundle_<v> <v>        # lite, qlora, plain
python deepseek_reference_logits.py bundle_<v> 12 <v>/reference.json 5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21
```

`eos_token_id` was then set to 63 in `config.json` and
`generation_config.json`, a token none of the references generates, so
generation runs the 12 steps. The weights and references do not depend on it.
Each directory keeps `config.json`, `model.safetensors`, `tokenizer.json` and
`tokenizer_config.json`.
