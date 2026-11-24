"""Run the pinned upstream Qwen3.5 gated attention block on synthetic weights.

Usage: python generate-qwen35-attention-reference.py MODELING_QWEN3_5.py > qwen35-attention-reference.json
Requires PyTorch. The fixture contains synthetic weights only.

MODELING_QWEN3_5.py is src/transformers/models/qwen3_5/modeling_qwen3_5.py at
Transformers revision 2f1bc5e321b9a736200a590d6e7743ccdeb60772 (Apache-2.0).
"""

import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn as nn

DIGEST = "c4d847b7e809ed2a49611e5afcea8bd0db39de597b67b57ce9753fb65f550de2"
data = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(data).hexdigest() != DIGEST:
    raise ValueError("unexpected upstream digest")
tree = ast.parse(data)

functions = {"rotate_half", "apply_rotary_pos_emb", "repeat_kv", "eager_attention_forward"}
classes = {"Qwen3_5RMSNorm", "Qwen3_5Attention", "Qwen3_5TextRotaryEmbedding"}
nodes = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in functions:
        node.decorator_list = []
        nodes.append(node)
    if isinstance(node, ast.ClassDef) and node.name in classes:
        node.decorator_list = []
        for method in node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Keep @staticmethod; drop the hub, rope-update and kwarg ones.
                method.decorator_list = [
                    d for d in method.decorator_list
                    if isinstance(d, ast.Name) and d.id == "staticmethod"
                ]
        nodes.append(node)
if {node.name for node in nodes} != functions | classes:
    raise ValueError("missing upstream attention definition")
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))


class Interfaces:
    @staticmethod
    def get_interface(name, default):
        if name != "eager":
            raise ValueError(name)
        return default


context = {"torch": torch, "nn": nn, "Callable": Callable,
           "ALL_ATTENTION_FUNCTIONS": Interfaces, "ROPE_INIT_FUNCTIONS": {}}
exec(compile(module, "pinned_qwen35_attention.py", "exec"), context)


class Cache:
    """The cache's one method the attention block calls: append K/V."""

    def __init__(self):
        self.k = self.v = None

    def update(self, k, v, layer_idx, **kwargs):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=-2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=-2)
        return self.k, self.v


def pattern(name, shape):
    seed = sum((i + 1) * ord(c) for i, c in enumerate(name)) % 97
    values = [((i * 17 + seed) % 101 - 50) / 113.0 for i in range(math.prod(shape))]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def flat(tensor):
    return tensor.flatten().tolist()


# 2 query heads of 8 over 1 KV head; a quarter of each head rotates, like
# Qwen3.5-0.8B's 256-wide heads with partial_rotary_factor 0.25.
cfg = SimpleNamespace(
    hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    attention_bias=False, attention_dropout=0.0, rms_norm_eps=1e-6,
    _attn_implementation="eager", max_position_embeddings=64,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
)
attention = context["Qwen3_5Attention"](cfg, 0).eval()
rotary = context["Qwen3_5TextRotaryEmbedding"](cfg)
with torch.no_grad():
    for name, parameter in attention.named_parameters():
        parameter.copy_(pattern(name, parameter.shape))

tokens = 5
hidden = pattern("hidden", (1, tokens, cfg.hidden_size))


def run(start, count, cache):
    positions = torch.arange(start, start + count).unsqueeze(0)
    cos, sin = rotary(hidden, positions)
    keys = start + count
    mask = torch.full((count, keys), float("-inf"))
    for row in range(count):
        mask[row, : start + row + 1] = 0.0
    output, _ = attention(hidden[:, start:start + count], (cos, sin),
                          mask.reshape(1, 1, count, keys), cache=cache)
    return output


with torch.no_grad():
    full = run(0, tokens, Cache())
    cache = Cache()
    steps = [run(0, 3, cache), run(3, 1, cache), run(4, 1, cache)]
torch.testing.assert_close(torch.cat(steps, 1), full, atol=1e-6, rtol=1e-6)
rotary_dim = rotary.inv_freq.shape[0] * 2
json.dump({
    "hidden_size": cfg.hidden_size, "tokens": tokens,
    "heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads,
    "head_dim": cfg.head_dim, "rotary_dim": rotary_dim,
    "rope_theta": cfg.rope_parameters["rope_theta"], "epsilon": cfg.rms_norm_eps,
    "hidden": flat(hidden),
    "weights": {name: flat(parameter) for name, parameter in attention.named_parameters()},
    "output": flat(full),
}, sys.stdout, indent=1)
print()
