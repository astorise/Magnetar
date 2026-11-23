"""Generate a token-shift reference from the pinned Olala attention class.

Usage: python reference.py /path/to/modeling_olala.py
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.nn.functional as F
from torch import nn

SOURCE_HASH = "357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf"


def upstream(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("unexpected upstream source digest")
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.ClassDef)
             and node.name in {"OlalaLinear", "OlalaDifferentialTensorProductAttentionV2"}]
    if len(nodes) != 2:
        raise ValueError("missing upstream classes")
    attention = next(node for node in nodes if node.name == "OlalaDifferentialTensorProductAttentionV2")
    forward = next(node for node in attention.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    # Return at the boundary immediately before Q/K normalization. This keeps
    # the upstream tensor-product projections and token-shift code unchanged.
    boundary = next(index for index, node in enumerate(forward.body)
                    if isinstance(node, ast.If) and ast.unparse(node.test) == "self.qk_norm")
    forward.body = forward.body[:boundary] + ast.parse("return key_states, value_states").body
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    namespace = {"torch": torch, "nn": nn, "F": F, "ATTN_IMPL": "eager"}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["OlalaDifferentialTensorProductAttentionV2"]


class Cache:
    def __init__(self):
        self.key = None
        self.value = None

    def get_last_kv(self, layer):
        assert layer == 0
        return self.key, self.value

    def set_last_kv(self, layer, key, value):
        assert layer == 0
        self.key, self.value = key.clone(), value.clone()


def flat(tensor):
    return tensor.detach().reshape(-1).tolist()


def main(path):
    torch.manual_seed(22751)
    config = SimpleNamespace(num_attention_heads=8, num_signal_heads_diff=6,
                             hidden_size=5, head_dim=4, tpa_rank=3,
                             qk_norm=False, softcap_attn=5.0, scalable_softmax=False,
                             cosnet=False, token_shift_attn=True,
                             scalar_proj_as_hidden_matrix=False, rope_theta=0.0,
                             rope_type="")
    model = upstream(path)(config, layer_idx=0).float().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.7)
        hidden = torch.randn(1, 7, config.hidden_size)
        positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])
        n, rank, dim = model.num_key_value_heads, model.rank, model.head_dim
        a_k = model.W_A_k(hidden).view(1, 7, n, rank)
        a_v = model.W_A_v(hidden).view(1, 7, n, rank)
        b_k = model.W_B_k(hidden).view(1, 7, rank, dim)
        b_v = model.W_B_v(hidden).view(1, 7, rank, dim)
        raw_k = torch.matmul(a_k, b_k) / rank
        raw_v = torch.matmul(a_v, b_v) / rank
        logits_k, logits_v = model.shift_proj_k(hidden), model.shift_proj_v(hidden)
        full_k, full_v = model(hidden, position_ids=positions)
        cache = Cache()
        chunks = []
        for start, end in [(0, 2), (2, 5), (5, 6), (6, 7)]:
            key, value = model(hidden[:, start:end], position_ids=positions[:, start:end], cache_params=cache)
            chunks.append({"start": start, "end": end, "key": flat(key), "value": flat(value),
                           "last_key": flat(cache.key), "last_value": flat(cache.value)})
        fixture = {"tokens": 7, "heads": n, "head_dim": dim,
                   "document_start": flat((positions == 0).float()),
                   "raw_key": flat(raw_k), "raw_value": flat(raw_v),
                   "key_logits": flat(logits_k), "value_logits": flat(logits_v),
                   "full_key": flat(full_k), "full_value": flat(full_v), "chunks": chunks}
    destination = Path(__file__).resolve().parents[3] / "magnetar-runtime/fixtures/olala-token-shift-reference.json"
    destination.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main(sys.argv[1])
