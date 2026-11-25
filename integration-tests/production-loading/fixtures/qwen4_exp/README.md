# Tiny Qwen4-Exp checkpoint and its `transformers` logits

Used by `src/tests_qwen4_exp_logits.rs` (`support-qwen38-flash-next`).
`dense/`, `sigmoid/` and `ple/` hold Hugging Face bundles (`qwen4_exp_text`), each with its `reference.json`; they differ in the linear layers' output gate (`silu`, `sigmoid`), an n-gram embedding in layer 0 (`ple`) and weights.
The reference holds the logits that `transformers` 5.23's
`Qwen4ExpForCausalLM` computes in float32
(`magnetar.inference-qualification.logit-reference/v1`), greedy, with the
whole sequence recomputed at every step (no cache).

The checkpoint is **random**, with weights scaled so that every mechanism
changes the logits:

| Part | Geometry |
| --- | --- |
| Model | 4 layers, hidden 32, vocabulary 64, untied embeddings |
| Residual | 4 streams (`hc_count`), input mixer of rank 6, no decoder norms |
| Layers 0 and 2 | gated-delta linear attention: 2 key heads and 4 value heads of 8, convolution kernel 4, `silu` output gate |
| Layers 1 and 3 | gated query attention: 2 query heads and 1 KV head of 16, a quarter of each head rotating (base 100) |
| Indexer | 2 heads and one key head of 8, budget 32 in blocks of 4 |
| MLP | every layer: 4 experts (top 2, renormalized) of width 12, and a shared expert of width 12 behind a sigmoid gate |
| Norms | zero-centered (scale `1 + weight`), except the mixers' gated norms |

The indexer keeps every key up to `budget + ratio - 1 = 35` visible tokens, so
attending densely gives its result: the prompt and the 12 generated tokens
(30 in all) stay within that. The Component refuses a graph beyond that many
tokens until the selection is built. Only `ple` has an n-gram embedding (layer 0: 16 hashed heads of 2 values, kernel 4, dilation 3, a table of 1,176 rows stored in four shards).

`transformers` 5.23.0's `modeling_qwen4_exp.py` is the source pinned by
design D7 of `support-qwen38-flash-next`. The reference uses no cache.

## How it was made

With the two scripts here: Python with torch 2.15.0 (CPU), `transformers`
5.23.0 and `safetensors`, the prompt
`5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21`, and 12 greedy steps:

    python make_tiny_qwen4_exp.py dense dense
    python qwen4_exp_reference_logits.py dense 12 dense/reference.json 5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21
    python make_tiny_qwen4_exp.py sigmoid sigmoid
    python qwen4_exp_reference_logits.py sigmoid 12 sigmoid/reference.json 5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21
    python make_tiny_qwen4_exp.py ple ple
    python qwen4_exp_reference_logits.py ple 12 ple/reference.json 5 17 33 8 41 3 60 12 9 27 44 19 7 52 30 11 38 21

`generation_config.json` is removed after saving: the Runtime does not read
it. The `dense` reference generates `16 28 42 51 51 51 51 11 10 38 40 28`, with
every step's top-two margin at least 0.089; the `sigmoid` one generates
`37 14 55 2 51 23 40 27 11 61 2 51` (margin at least 0.019) and the `ple` one
`35 25 35 25 51 45 11 34 44 51 45 39` (margin at least 0.043).

On the Reference CPU Provider the largest logit difference is 1.7e-6 (`dense`)
2.1e-6 (`sigmoid`) and 3.0e-7 (`ple`), and on the optimized CPU Provider 3.7e-6, 2.1e-6
and 1.8e-6.
