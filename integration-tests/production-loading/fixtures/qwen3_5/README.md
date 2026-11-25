# Tiny Qwen3.5 checkpoint and its `transformers` logits

Used by `src/tests_qwen3_5_logits.rs` (`support-qwen38-flash-next`).
`hybrid/` holds a Hugging Face bundle (`qwen3_5_text`) and `reference.json`.
The reference holds the logits that `transformers` 5.23's
`Qwen3_5ForCausalLM` computes in float32
(`magnetar.inference-qualification.logit-reference/v1`), greedy, with the
whole sequence recomputed at every step (no cache).

The checkpoint is **random**, with weights scaled so that every mechanism
changes the logits:

| Part | Geometry |
| --- | --- |
| Model | 4 layers, hidden 32, vocabulary 64, tied embeddings |
| Layers 0 and 2 | gated-delta linear attention: 2 key heads and 4 value heads of 8, convolution kernel 4 |
| Layers 1 and 3 | gated full attention: 2 query heads and 1 KV head of 16, a quarter of each head rotating (base 100) |
| Norms | zero-centered (scale `1 + weight`), except the mixers' gated norms |

`transformers` 5.23.0's `modeling_qwen3_5.py` differs from the source
pinned by the Runtime's fixtures (revision 2f1bc5e) only in the name of
the cache argument. The reference uses no cache.

## How it was made

With the two scripts here: Python with torch 2.15.0 (CPU), `transformers`
5.23.0 and `safetensors`, the prompt
`5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21`, and 12 greedy steps:

    python make_tiny_qwen3_5.py hybrid hybrid
    python qwen3_5_reference_logits.py hybrid 12 hybrid/reference.json 5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21

`generation_config.json` is removed after saving: the Runtime does not read
it. The reference generates `57 3 3 3 3 35 47 47 47 47 48 48`, with every
step's top-two margin at least 0.49.
