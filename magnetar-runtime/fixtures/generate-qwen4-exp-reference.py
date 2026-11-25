"""Run Transformers' `qwen4_exp` blocks on synthetic weights and print a fixture.

Usage: python generate-qwen4-exp-reference.py > qwen4-exp-reference.json
Requires PyTorch and Transformers 5.23.0, whose `models/qwen4_exp` (Apache-2.0)
this script checks by SHA-256 before importing it. The fixture holds synthetic
weights only. See design D7 of openspec/changes/support-qwen38-flash-next.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import transformers

PINNED = {
    "modular_qwen4_exp.py": "a5350d43d3bd8169c566200ad7efc87580d39da0371eeaf3b3e14e6f9696dde2",
    "modeling_qwen4_exp.py": "6bba060180c7f3c9d4d208941bbc2dfffb8933698ca332ef86179ce72355b3c0",
    "configuration_qwen4_exp.py": "8dfb0ef09612a7eea896672798d9a3e735b1d00e0caf935072d2cffe5afce9ab",
}
source = Path(transformers.__file__).parent / "models" / "qwen4_exp"
for name, digest in PINNED.items():
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
        sys.exit(f"{name} does not match the pinned Transformers 5.23.0 source")

from transformers.models.qwen4_exp import modeling_qwen4_exp as m  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

torch.manual_seed(20260925)
HIDDEN, HC, RANK, TOKENS = 8, 4, 6, 3
config = Qwen4ExpTextConfig(
    hidden_size=HIDDEN, hc_count=HC, hc_lowrank=RANK, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=4, linear_value_head_dim=4,
    moe_intermediate_size=6, shared_expert_intermediate_size=6,
    num_experts=4, num_experts_per_tok=2, vocab_size=32, rms_norm_eps=1e-6,
)


def flat(tensor):
    return {"shape": list(tensor.shape), "data": tensor.detach().flatten().tolist()}


def randomize(module):
    # Every weight non-trivial, including the norm's offset-by-one weight.
    for parameter in module.parameters():
        parameter.data = torch.randn_like(parameter) * 0.7


fixture = {"source": "transformers 5.23.0 models/qwen4_exp", "hidden": HIDDEN,
           "hc_count": HC, "lowrank": RANK, "epsilon": config.rms_norm_eps}

# GatedResidual with and without the injection weights.
residual = m.Qwen4ExpTextGatedResidual(config)
randomize(residual)
h = torch.randn(TOKENS, HC * HIDDEN) * 1.3
with torch.no_grad():
    mixed, hyper_input, injection = residual(h)
    # The decoder layer's closing step, for an arbitrary block output.
    block_out = torch.randn(TOKENS, HIDDEN)
    inj = block_out.unsqueeze(-2) * injection.unsqueeze(-1)
    after = hyper_input + inj.flatten(-2)
    final = m.Qwen4ExpTextGatedResidual(config, use_combine=False)
    final.load_state_dict({k: v for k, v in residual.state_dict().items() if "inject" not in k})
    final_mixed = final(h)
    embedding = torch.randn(TOKENS, HIDDEN)
fixture["gated_residual"] = {
    "norm_weight": flat(residual.hc_norm.weight),
    "down": flat(residual.input_mix_weight_down.weight),
    "up": flat(residual.input_mix_weight_up.weight),
    "inject": flat(residual.block_inject_weight.weight),
    "input": flat(h), "mixed": flat(mixed), "injection": flat(injection),
    "block_output": flat(block_out), "after_block": flat(after),
    "final_mixed": flat(final_mixed),
    "embedding": flat(embedding), "repeated": flat(embedding.repeat(1, HC)),
}
# The QSA indexer, on a configuration where the selection prunes: budget 8
# tokens in blocks of 2, over 20 tokens. Rotary dimension 4 of 8.
INDEX_HEADS, INDEX_DIM, BUDGET, RATIO, QSA_TOKENS = 4, 8, 8, 2, 20
qsa_config = Qwen4ExpTextConfig(
    hidden_size=HIDDEN, hc_count=HC, hc_lowrank=RANK, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=4, linear_value_head_dim=4,
    moe_intermediate_size=6, shared_expert_intermediate_size=6,
    num_experts=4, num_experts_per_tok=2, vocab_size=32, rms_norm_eps=1e-6,
    indexer_n_heads=INDEX_HEADS, indexer_kv_heads=1, indexer_head_dim=INDEX_DIM,
    indexer_budget=BUDGET, indexer_compress_ratio=RATIO,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0],
                     "mrope_interleaved": True},
)


def block_scores(q, keys, norm_weight, query_index):
    """The indexer's block scores for one query, by the module's formulas.

    Used only to choose a seed whose top-k has no tie at its cut: torch.topk
    does not define ties, and the relu makes zero scores common. The selection
    in the fixture always comes from running the module.
    """
    rotary_dim, base = 4, 10000.0
    scores = []
    for block in range((query_index + 1) // RATIO):
        pooled = keys[block * RATIO:(block + 1) * RATIO].mean(0)
        pooled = pooled * torch.rsqrt(pooled.pow(2).mean() + 1e-6) * (1 + norm_weight)
        half = rotary_dim // 2
        inv = 1.0 / base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
        angle = (block * RATIO) * inv
        cos, sin = torch.cos(angle), torch.sin(angle)
        a, b = pooled[:half].clone(), pooled[half:rotary_dim].clone()
        pooled = pooled.clone()
        pooled[:half], pooled[half:rotary_dim] = a * cos - b * sin, b * cos + a * sin
        scores.append(torch.relu(q[query_index] @ pooled).sum() / math.sqrt(INDEX_DIM))
    return torch.stack(scores) if scores else torch.zeros(0)


def has_tie_at_cut(q, keys, norm_weight):
    kept = BUDGET // RATIO
    for query_index in range(QSA_TOKENS):
        scores = block_scores(q, keys, norm_weight, query_index)
        if len(scores) > kept:
            ordered = scores.sort(descending=True).values
            if ordered[kept - 1] - ordered[kept] < 1e-4:
                return True
    return False


for attempt in range(200):
    indexer = m.Qwen4ExpTextQSAIndexer(qsa_config, 3)
    randomize(indexer)
    hidden_states = torch.randn(1, QSA_TOKENS, HIDDEN)
    position_ids = torch.arange(QSA_TOKENS).view(1, 1, -1).expand(3, 1, -1)
    rotary = m.Qwen4ExpTextRotaryEmbedding(qsa_config)
    with torch.no_grad():
        cos, sin = rotary(hidden_states, position_ids)
        qk = indexer.index_qk_proj(hidden_states)
        q, token_k = torch.split(qk, [INDEX_HEADS * INDEX_DIM, INDEX_DIM], dim=-1)
        q = indexer.q_layernorm(q.reshape(1, QSA_TOKENS, INDEX_HEADS, INDEX_DIM))
        q = m.apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2)
        keys = token_k.reshape(QSA_TOKENS, INDEX_DIM)
        if not has_tie_at_cut(q[0], keys, indexer.k_layernorm.weight):
            break
else:
    sys.exit("no seed without a tie at the top-k cut")
with torch.no_grad():
    causal = torch.tril(torch.ones(QSA_TOKENS, QSA_TOKENS, dtype=torch.bool))[None, None]
    selected = indexer(hidden_states, (cos, sin), causal, None)
fixture["qsa_indexer"] = {
    "head_count": INDEX_HEADS, "head_dim": INDEX_DIM, "budget": BUDGET,
    "compress_ratio": RATIO, "rotary_dim": int(cos.shape[-1]),
    "rope_base": 10000.0,
    "key_norm_weight": flat(indexer.k_layernorm.weight),
    "queries": flat(q.reshape(QSA_TOKENS, INDEX_HEADS * INDEX_DIM)),
    "keys": flat(keys),
    "selected": {"shape": list(selected.shape[-2:]),
                 "data": selected[0, 0].flatten().to(torch.float32).tolist()},
}
# The hashed n-gram embedding and the PLE layer. Layer index 2 is the second
# PLE layer (ple_layer_index 1), so the head vocabularies and multipliers are
# not the first layer's. Token 7 is the EOS: it ends an n-gram segment.
from transformers.cache_utils import DynamicCache  # noqa: E402

PLE_EMBED, NGRAM_SIZE, HEADS_PER_NGRAM, VOCAB, EOS = 8, 3, 2, 32, 7
ple_config = Qwen4ExpTextConfig(
    hidden_size=HIDDEN, hc_count=HC, hc_lowrank=RANK, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=4, linear_value_head_dim=4,
    moe_intermediate_size=6, shared_expert_intermediate_size=6,
    num_experts=4, num_experts_per_tok=2, vocab_size=VOCAB, rms_norm_eps=1e-6,
    ple_layer_ids=[1, 2], ple_embed_dim=PLE_EMBED, ple_conv_kernel_size=3,
    ngram_size=NGRAM_SIZE, heads_per_ngram=HEADS_PER_NGRAM,
    ngram_vocab_size_base=50, make_ngram_vocab_size_divisible_by=4,
    seed=1234, eos_token_id=EOS,
)
LAYER, PLE_INDEX = 1, 1
ple = m.Qwen4ExpTextPLELayer(ple_config, LAYER, PLE_INDEX)
randomize(ple)
ple.conv1d.weight.data *= 0.5
PLE_TOKENS = 12
token_ids = torch.tensor([[3, 9, 7, 12, 12, 30, 5, 7, 7, 21, 4, 18]])
ple_hidden = torch.randn(1, PLE_TOKENS, HC * HIDDEN)
with torch.no_grad():
    embeddings = ple.ple_embedding(token_ids, None)
    ple_out = ple(ple_hidden, token_ids, None)
    # The same sequence in two chunks, the cache carrying both states.
    cache = DynamicCache(config=ple_config)
    first = ple(ple_hidden[:, :5], token_ids[:, :5], cache)
    second = ple(ple_hidden[:, 5:], token_ids[:, 5:], cache)
    chunked = torch.cat([first, second], dim=1)
table = ple.ple_embedding.ngram_embedding.weight
fixture["ngram_ple"] = {
    "ngram_size": NGRAM_SIZE, "heads_per_ngram": HEADS_PER_NGRAM,
    "vocab_size": VOCAB, "vocab_base": 50, "layer_index": PLE_INDEX,
    "seed": 1234, "eos_token_id": EOS,
    "conv_kernel": 3, "conv_dilation": NGRAM_SIZE,
    "token_ids": [float(t) for t in token_ids[0].tolist()],
    "table": flat(table),
    "head_vocab_sizes": [int(v) for v in ple.ple_embedding.head_vocab_sizes],
    "multipliers": [int(v) for v in ple.ple_embedding.layer_multipliers],
    "embeddings": flat(embeddings[0]),
    "hidden": flat(ple_hidden[0]),
    "key_proj": flat(ple.key_proj.weight), "value_proj": flat(ple.value_proj.weight),
    "norm_key": flat(ple.norm_key.weight), "norm_query": flat(ple.norm_query.weight),
    "norm_conv": flat(ple.norm_conv.weight),
    "conv": flat(ple.conv1d.weight.squeeze(1)),
    "output": flat(ple_out[0]),
    "chunked_output": flat(chunked[0]),
}
json.dump(fixture, sys.stdout, indent=1)
print()
