"""Capture the pinned Olala attention core at its actual eager boundary.

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
    names = {"OlalaLinear", "OlalaNorm", "OlalaRMSNorm", "eager_attention_forward",
             "OlalaDifferentialTensorProductAttentionV2"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("missing upstream definitions")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    namespace = {"torch": torch, "nn": nn, "F": F, "ATTN_IMPL": "eager"}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class Cache:
    def __init__(self):
        self.key = self.value = self.raw_key = self.raw_value = None

    def get_last_kv(self, layer):
        assert layer == 0
        return self.raw_key, self.raw_value

    def set_last_kv(self, layer, key, value):
        assert layer == 0
        self.raw_key, self.raw_value = key.clone(), value.clone()

    def update(self, key, value, layer):
        assert layer == 0
        self.key = key if self.key is None else torch.cat([self.key, key], dim=1)
        self.value = value if self.value is None else torch.cat([self.value, value], dim=1)
        return self.key, self.value


def flat(tensor):
    return tensor.detach().float().reshape(-1).tolist()


def main(path):
    torch.manual_seed(56731)
    config = SimpleNamespace(num_attention_heads=8, num_signal_heads_diff=6,
                             hidden_size=5, head_dim=4, tpa_rank=3,
                             qk_norm=True, normalization_type="rmsnorm", norm_epsilon=1e-6,
                             zero_centered_gamma=True, softcap_attn=1.4, scalable_softmax=True,
                             cosnet=False, token_shift_attn=True, scalar_proj_as_hidden_matrix=False,
                             rope_theta=0.0, rope_type="", slw_wsize=3, use_completed_p=False,
                             intra_doc_masking=False, xsa=False)
    namespace = upstream(path)
    capture = {}
    eager = namespace["eager_attention_forward"]

    def wrapped(query, key, value, **kwargs):
        capture.update(query=flat(query), key=flat(key), value=flat(value),
                       key_tokens=key.shape[1])
        return eager(query, key, value, **kwargs)

    namespace["eager_attention_forward"] = wrapped
    model = namespace["OlalaDifferentialTensorProductAttentionV2"](config, layer_idx=0).float().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn_like(parameter) * (0.15 if "norm" in name else 0.7))
        hidden = torch.randn(1, 7, config.hidden_size)
        positions = torch.arange(7).view(1, 7)

        def run(start, end, cache):
            capture.clear()
            part = hidden[:, start:end]
            output = model(part, position_ids=positions[:, start:end], cache_params=cache)[0]
            return {**capture, "query_tokens": end - start,
                    "lambda_logits": flat(model.lambda_proj(part)), "output": flat(output)}

        full = run(0, 7, None)
        cache = Cache()
        steps = [run(start, end, cache) for start, end in [(0, 2), (2, 4), (4, 5), (5, 7)]]
        fixture = {"query_heads": model.num_attention_heads, "kv_heads": model.num_key_value_heads,
                   "signal_heads": model.num_signal_heads, "head_dim": config.head_dim,
                   "window_left": config.slw_wsize, "softcap": config.softcap_attn,
                   "scale": config.head_dim ** -0.5, "full": full, "steps": steps}
    destination = Path(__file__).resolve().parents[3] / "magnetar-runtime/fixtures/olala-differential-attention-reference.json"
    destination.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main(sys.argv[1])
