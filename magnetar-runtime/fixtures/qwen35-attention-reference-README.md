# Qwen3.5 gated attention fixture

`qwen35-attention-reference.json` holds synthetic weights only. It is the
output of the upstream `Qwen3_5Attention` block (Apache-2.0) at Transformers
revision
[2f1bc5e321b9a736200a590d6e7743ccdeb60772](https://github.com/huggingface/transformers/blob/2f1bc5e321b9a736200a590d6e7743ccdeb60772/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
whose source SHA-256 is
`c4d847b7e809ed2a49611e5afcea8bd0db39de597b67b57ce9753fb65f550de2`.

`generate-qwen35-attention-reference.py` checks that digest. It then runs
these upstream definitions with PyTorch, and nothing else:

- `Qwen3_5Attention`, `Qwen3_5RMSNorm` and `Qwen3_5TextRotaryEmbedding`;
- `rotate_half`, `apply_rotary_pos_emb`, `repeat_kv` and `eager_attention_forward`.

The block has 2 query heads of 8 over 1 KV head, with half of each head
rotating. It is fed 5 synthetic tokens and a causal mask. The generator also
checks that a 3-token prefill followed by two cached decode steps equals the
5-token prefill.

To regenerate:

    python generate-qwen35-attention-reference.py MODELING_QWEN3_5.py > qwen35-attention-reference.json

`qwen35_gated_attention_block_matches_pinned_reference` builds the same block
from portable operators, as the Qwen Component emits it:

- a per-head `split` of the query and gate;
- per-head `rmsnorm` with `weight_offset` 1;
- a half-split `rope` over the rotary dimension;
- grouped `attention`;
- a sigmoid `activation` and `mul` for the output gate.

It compares the output on the Reference CPU. The test does not cover
checkpoint weights, BF16 or KV history across steps.
