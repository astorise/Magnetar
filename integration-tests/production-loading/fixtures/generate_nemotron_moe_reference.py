"""Generate a tiny Nemotron-H MoE fixture from NVIDIA's pinned PyTorch code.

Usage: python generate_nemotron_moe_reference.py MODELING.py OUTPUT.json
Only the three upstream MoE classes are executed; the model source is checked
against its pinned digest. Parameters and inputs are deterministic synthetic
values and no checkpoint weights are distributed.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

source_path, output_path = map(Path, sys.argv[1:])
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "52273a23a2c768ffe5938e96f6832afd762029293a8be9bfed886757ba6a39b5"
names = {"NemotronHMLP", "NemotronHMOE", "NemotronHTopkRouter"}
classes = [item for item in ast.parse(source).body if isinstance(item, ast.ClassDef) and item.name in names]
assert {item.name for item in classes} == names
module = ast.Module(body=ast.parse("from __future__ import annotations").body + classes, type_ignores=[])
scope = {"torch": torch, "nn": nn, "F": F, "ACT2FN": ACT2FN,
         "logger": SimpleNamespace(warning_once=lambda *_: None)}
exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), scope)


def tensor(value):
    value = value.detach().contiguous()
    return {"shape": list(value.shape), "data": value.reshape(-1).tolist()}


cases = []
for top_k in (1, 2):
    torch.manual_seed(9102)
    config = SimpleNamespace(hidden_size=5, intermediate_size=3, mlp_bias=False,
                             mlp_hidden_act="relu2", n_routed_experts=4,
                             moe_intermediate_size=3, moe_shared_expert_intermediate_size=6,
                             num_experts_per_tok=top_k, routed_scaling_factor=2.5,
                             n_group=1, topk_group=1, norm_topk_prob=True)
    model = scope["NemotronHMOE"](config, layer_idx=1).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.3)
        model.gate.e_score_correction_bias.copy_(torch.tensor([-0.45, 0.35, 0.1, -0.15]))
        hidden = torch.randn(4, config.hidden_size) * 0.6
        head = torch.randn(config.hidden_size, 7) * 0.2
        selected, weights = model.gate(hidden)
        unbiased = torch.sigmoid(F.linear(hidden, model.gate.weight)).topk(top_k, dim=-1).indices
        if not any(set(a.tolist()) != set(b.tolist()) for a, b in zip(selected, unbiased)):
            raise AssertionError("the correction bias must change at least one selected set")
        output = model(hidden)
        values = {
            "hidden": hidden,
            "router": model.gate.weight.T,
            "expert_in": torch.cat([expert.up_proj.weight.T for expert in model.experts], dim=1),
            "expert_out": torch.cat([expert.down_proj.weight.T for expert in model.experts], dim=1),
            "bias": model.gate.e_score_correction_bias,
            "shared_in": model.shared_experts.up_proj.weight.T,
            "shared_out": model.shared_experts.down_proj.weight.T,
            "head": head,
            "output": output,
            "logits": output @ head,
        }
        cases.append({"top_k": top_k, "scale": config.routed_scaling_factor,
                      "experts": config.n_routed_experts,
                      "selected": selected.tolist(), "weights": weights.tolist(),
                      "tensors": {name: tensor(value) for name, value in values.items()}})
output_path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
