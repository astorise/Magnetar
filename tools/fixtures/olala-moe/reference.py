"""Make tiny full OlalaMoE references from the pinned upstream PyTorch code.

Usage: python reference.py path/to/modeling_olala.py
The verified upstream OlalaMoE, TorchMoEMLP, _TorchParallelExperts and
OlalaMLP methods execute on synthetic, deterministic weights. No checkpoint
weights are redistributed.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

SOURCE_HASH = "357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf"


def upstream(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("unexpected Olala source digest")
    tree = ast.parse(source)
    names = {"OlalaMoE", "TorchMoEMLP", "_TorchParallelExperts", "OlalaMLP"}
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("missing upstream classes")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def tensor(value):
    return {"shape": list(value.shape), "data": value.detach().reshape(-1).tolist()}


def case(classes, top_k):
    torch.manual_seed(71345)
    tokens, hidden, bottleneck, intermediate, shared, experts, vocabulary = 4, 5, 3, 4, 6, 4, 7
    model = classes["OlalaMoE"].__new__(classes["OlalaMoE"])
    nn.Module.__init__(model)
    model.config = SimpleNamespace(moe_num_active_experts=top_k, moe_routed_scaling_factor=2.45,
                                   moe_routed_input_dim=bottleneck, moe_shared_expert_gate=False)
    model.moe_gate = nn.Linear(hidden, experts, bias=False)
    model.expert_bias = torch.tensor([0.15, -0.25, 0.35, -0.1])
    model.down_proj = nn.Linear(hidden, bottleneck, bias=False)
    model.up_proj = nn.Linear(bottleneck, hidden, bias=False)
    model.experts = classes["TorchMoEMLP"](bottleneck, intermediate, experts, top_k,
                                           activation=lambda x: F.relu(x).square())
    model.shared_experts = classes["OlalaMLP"].__new__(classes["OlalaMLP"])
    nn.Module.__init__(model.shared_experts)
    model.shared_experts.fc_1 = nn.Linear(hidden, shared, bias=False)
    model.shared_experts.fc_2 = nn.Linear(shared, hidden, bias=False)
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.45)
        inputs = torch.randn(tokens, hidden) * 0.7
        head = torch.randn(hidden, vocabulary) * 0.5
        output = model(inputs.reshape(1, tokens, hidden)).reshape(tokens, hidden)
        logits = output @ head
    values = {
        "hidden": inputs,
        "router": model.moe_gate.weight.t().contiguous(),
        "routed_in": model.down_proj.weight.t().contiguous(),
        "expert_in": model.experts.experts.weight.permute(2, 0, 1).contiguous().reshape(bottleneck, -1),
        "expert_out": model.experts.output_experts.weight.permute(2, 0, 1).contiguous().reshape(intermediate, -1),
        "bias": model.expert_bias,
        "routed_out": model.up_proj.weight.t().contiguous(),
        "shared_in": model.shared_experts.fc_1.weight.t().contiguous(),
        "shared_out": model.shared_experts.fc_2.weight.t().contiguous(),
        "head": head,
        "output": output,
        "logits": logits,
    }
    return {"top_k": top_k, "scale": 2.45, "experts": experts,
            "tensors": {name: tensor(value) for name, value in values.items()}}


if __name__ == "__main__":
    classes = upstream(sys.argv[1])
    destination = Path(__file__).resolve().parents[3] / "magnetar-runtime/fixtures/olala-moe-reference.json"
    destination.write_text(json.dumps([case(classes, 1), case(classes, 2)], indent=2) + "\n", encoding="utf-8")
    print(destination)
