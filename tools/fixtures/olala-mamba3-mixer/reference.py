"""Run the pinned Olala Mamba-3 layer with small synthetic CPU weights.

Usage: python reference.py /path/to/modeling_olala.py
"""
import ast
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

SOURCE_HASH = "357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf"


def upstream(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("unexpected upstream source digest")
    names = {"OlalaNorm", "OlalaRMSNorm", "OlalaLinear", "OlalaMamba3MimoFast",
             "mamba3_tilelang", "_mamba3_rotate", "_mamba3_mimo_torch_segment",
             "_mamba3_mimo_torch", "_mamba3_rotary_step_torch", "_mamba3_step_torch",
             "_pack_cu_seqlens"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("missing upstream classes or functions")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math, "os": os,
                 "rearrange": rearrange, "nullcontext": nullcontext,
                 "_mamba3_mimo": None, "mamba3_step_fn": None}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["OlalaMamba3MimoFast"]


def flat(value):
    return value.detach().reshape(-1).tolist()


def state(cache):
    return {"angle": flat(cache.mamba3_angle_states[0]),
            "ssm": flat(cache.mamba3_ssm_states[0]),
            "key": flat(cache.mamba3_k_states[0]),
            "input": flat(cache.mamba3_v_states[0])}


def main(path):
    torch.manual_seed(44991)
    config = SimpleNamespace(hidden_size=4, mamba_mimo_dim=2, mamba_d_state=4,
                             mamba_headdim=2, mamba_ngroups=1,
                             normalization_type="rmsnorm", norm_epsilon=1e-6,
                             zero_centered_gamma=True, cosnet=False,
                             mamba3_postgate_norm=False, complete_slw=False, slw_wsize=128)
    model = upstream(path)(config, layer_idx=0).float().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn_like(parameter) * (0.15 if "norm" in name else 0.3))
        inputs = torch.randn(1, 6, config.hidden_size) * 0.7
        full = model(inputs)[0]
        cache = SimpleNamespace(mamba3_angle_states=[None], mamba3_ssm_states=[None],
                                mamba3_k_states=[None], mamba3_v_states=[None], past_length=[0])
        prefix = model(inputs[:, :3], cache_params=cache)[0]
        prefix_state = state(cache)
        steps = []
        for index in range(3, 6):
            output = model(inputs[:, index:index + 1], cache_params=cache)[0]
            steps.append({"output": flat(output), **state(cache)})
    parameter = lambda name: dict(model.named_parameters())[name]
    fixture = {
        "geometry": {"hidden": config.hidden_size, "heads": model.nheads,
                     "groups": model.ngroups, "rank": model.mimo_dim,
                     "head_dim": model.headdim, "state_dim": model.d_state,
                     "angles": model.num_rope_angles, "epsilon": config.norm_epsilon,
                     "decay_floor": model.A_floor, "zero_centered_norm": True},
        "inputs": flat(inputs),
        "weights": {
            "in_proj": flat(parameter("in_proj.weight").t()),
            "in_proj_dyn": flat(parameter("in_proj_dyn.weight").t()),
            "dt_bias": flat(parameter("dt_bias")),
            "b_norm": flat(parameter("B_norm.norm.weight")),
            "c_norm": flat(parameter("C_norm.norm.weight")),
            "q_bias": flat(parameter("C_bias").transpose(0, 1)),
            "k_bias": flat(parameter("B_bias").transpose(0, 1)),
            "x_proj": flat(parameter("in_proj_mimo_x").transpose(0, 1)),
            "z_proj": flat(parameter("in_proj_mimo_z").transpose(0, 1)),
            "out_proj": flat(parameter("out_proj_mimo").transpose(0, 1)),
            "skip": flat(parameter("D")),
        },
        "full_output": flat(full), "prefix_output": flat(prefix),
        "prefix_state": prefix_state, "steps": steps,
    }
    destination = Path(__file__).resolve().parents[3] / "magnetar-runtime/fixtures/olala-mamba3-mixer-reference.json"
    destination.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main(sys.argv[1])
