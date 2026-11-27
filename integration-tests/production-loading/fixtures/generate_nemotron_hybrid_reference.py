"""Generate a tiny mixed Nemotron-H reference from pinned PyTorch primitives.

Usage: python generate_nemotron_hybrid_reference.py MODELING.py LAYERNORM.py OUTPUT.json
Both upstream sources are SHA-256 checked. The composition explicitly passes
attention KV state: the pinned NVIDIA block does not do that itself. All
weights and tokens are synthetic; this does not qualify the real checkpoint.
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
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN


def extract(path, digest, names, namespace):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise ValueError(f"unexpected source digest: {path}")
    nodes = [item for item in ast.parse(source).body
             if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name in names]
    if {item.name for item in nodes} != set(names):
        raise ValueError("missing upstream definitions")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class MambaCache:
    """Correct the upstream cache's list/.device mismatch, as in mixer fixtures."""
    def __init__(self, channels, kernel, heads, head_dim, state_dim):
        self.conv_kernel_size = kernel
        self.conv_states = torch.zeros(1, 1, channels, kernel)
        self.ssm_states = torch.zeros(1, 1, heads, head_dim, state_dim)

    def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(-1, -1)
            self.conv_states[layer_idx, :, :, -1] = new_conv_state[:, 0, :]

    def update_ssm_state(self, layer_idx, new_ssm_state):
        self.ssm_states[layer_idx] = new_ssm_state


class AttentionCache:
    def __init__(self):
        self.key = None
        self.value = None

    def update(self, key, value, layer_idx):
        assert layer_idx == 1
        self.key = key if self.key is None else torch.cat([self.key, key], dim=-2)
        self.value = value if self.value is None else torch.cat([self.value, value], dim=-2)
        return self.key, self.value


def tensor(value):
    value = value.detach().contiguous()
    return {"shape": list(value.shape), "data": value.reshape(-1).tolist()}


def main():
    if len(sys.argv) != 4:
        raise ValueError("expected MODELING.py LAYERNORM.py OUTPUT.json")
    scope = {"torch": torch, "nn": nn, "F": F, "rearrange": rearrange, "ACT2FN": ACT2FN,
             "is_fast_path_available": False, "logger": SimpleNamespace(warning_once=lambda *_: None)}
    extract(sys.argv[2], "eb6252e247b90f1c8a75946efbc1a221e0c4da701b6757ddae49f3495cf7a42f",
            ["rms_norm_ref"], scope)
    scope["rmsnorm_fn"] = scope["rms_norm_ref"]
    extract(sys.argv[1], "52273a23a2c768ffe5938e96f6832afd762029293a8be9bfed886757ba6a39b5",
            ["pad_tensor_by_size", "reshape_into_chunks", "segment_sum", "apply_mask_to_padding_states",
             "MambaRMSNormGated", "NemotronHMamba2Mixer", "NemotronHRMSNorm", "NemotronHMLP",
             "NemotronHMOE", "NemotronHTopkRouter", "repeat_kv", "NemotronHAttention"], scope)
    cfg = SimpleNamespace(hidden_size=5, intermediate_size=3, num_attention_heads=1,
                          num_key_value_heads=1, head_dim=5, max_position_embeddings=64,
                          attention_dropout=0.0, attention_bias=False, mamba_num_heads=4,
                          mamba_head_dim=2, ssm_state_size=3, n_groups=2, conv_kernel=3,
                          chunk_size=4, use_conv_bias=True, use_bias=False, mamba_hidden_act="silu",
                          layer_norm_epsilon=1e-5, time_step_limit=(0.0, float("inf")),
                          time_step_min=0.001, time_step_max=0.1, mlp_bias=False, mlp_hidden_act="relu2",
                          n_routed_experts=4, moe_intermediate_size=3, moe_shared_expert_intermediate_size=6,
                          num_experts_per_tok=2, routed_scaling_factor=2.5, n_group=1, topk_group=1,
                          norm_topk_prob=True)
    torch.manual_seed(49812)
    mixer = scope["NemotronHMamba2Mixer"](cfg, 0).eval()
    attention = scope["NemotronHAttention"](cfg, 1).eval()
    dense = scope["NemotronHMLP"](cfg, layer_idx=2).eval()
    moe = scope["NemotronHMOE"](cfg, 3).eval()
    norms = [scope["NemotronHRMSNorm"](cfg.hidden_size, eps=cfg.layer_norm_epsilon) for _ in range(5)]
    embedding = nn.Embedding(24, cfg.hidden_size)
    head = nn.Linear(cfg.hidden_size, 24, bias=False)
    with torch.no_grad():
        for model in [mixer, attention, dense, moe, *norms, embedding, head]:
            for name, parameter in model.named_parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.2)
                if name == "norm.weight" or model in norms:
                    parameter.add_(1.0)
        moe.gate.e_score_correction_bias.copy_(torch.tensor([-0.45, 0.35, 0.1, -0.15]))
        ids = torch.tensor([[2, 7, 3, 12, 1, 9, 18, 4]])

        def new_cache():
            return (MambaCache(mixer.conv_dim, cfg.conv_kernel, cfg.mamba_num_heads,
                               cfg.mamba_head_dim, cfg.ssm_state_size), AttentionCache())

        def run(tokens, start, cache):
            mamba_cache, attention_cache = cache
            positions = torch.arange(start, start + tokens.shape[1])
            hidden = embedding(tokens)
            hidden = hidden + mixer.torch_forward(norms[0](hidden), mamba_cache, positions)
            # The reference composition intentionally forwards the cache; the
            # pinned whole-model block omits it. Prefix is causal and cached
            # continuation consists only of single-token calls.
            hidden = hidden + attention(norms[1](hidden), past_key_value=attention_cache,
                                        cache_position=positions)[0]
            hidden = hidden + dense(norms[2](hidden))
            hidden = hidden + moe(norms[3](hidden))
            return head(norms[4](hidden))

        full_cache = new_cache()
        full = run(ids, 0, full_cache)
        cache = new_cache()
        prefix = run(ids[:, :4], 0, cache)
        prefix_state = {"conv": tensor(cache[0].conv_states), "ssm": tensor(cache[0].ssm_states),
                        "keys": tensor(cache[1].key), "values": tensor(cache[1].value)}
        steps = []
        for index in range(4, ids.shape[1]):
            output = run(ids[:, index:index+1], index, cache)
            steps.append({"logits": tensor(output), "conv": tensor(cache[0].conv_states),
                          "ssm": tensor(cache[0].ssm_states), "keys": tensor(cache[1].key),
                          "values": tensor(cache[1].value)})
        joined = torch.cat([prefix] + [torch.tensor(step["logits"]["data"]).view(1, 1, -1)
                                       for step in steps], dim=1)
        if not torch.allclose(joined, full, atol=3e-6, rtol=1e-6):
            raise AssertionError(f"prefill/decode disagree: {(joined - full).abs().max().item()}")
        weights = {
            "token_embedding": embedding.weight, "final_norm": norms[4].weight, "lm_head": head.weight.T,
            "layers.0.norm": norms[0].weight, "layers.0.mamba2.in_proj": mixer.in_proj.weight.T,
            "layers.0.mamba2.conv": mixer.conv1d.weight.reshape(mixer.conv_dim, cfg.conv_kernel),
            "layers.0.mamba2.dt_bias": mixer.dt_bias, "layers.0.mamba2.a_log": mixer.A_log,
            "layers.0.mamba2.d": mixer.D, "layers.0.mamba2.norm": mixer.norm.weight,
            "layers.0.mamba2.out_proj": mixer.out_proj.weight.T, "layers.0.mamba2.conv_bias": mixer.conv1d.bias,
            "layers.1.norm": norms[1].weight, "layers.1.attention.q_proj": attention.q_proj.weight.T,
            "layers.1.attention.k_proj": attention.k_proj.weight.T, "layers.1.attention.v_proj": attention.v_proj.weight.T,
            "layers.1.attention.o_proj": attention.o_proj.weight.T, "layers.2.norm": norms[2].weight,
            "layers.2.mlp.in": dense.up_proj.weight.T, "layers.2.mlp.out": dense.down_proj.weight.T,
            "layers.3.norm": norms[3].weight, "layers.3.moe.router": moe.gate.weight.T,
            "layers.3.moe.experts_in": torch.cat([expert.up_proj.weight.T for expert in moe.experts], dim=1),
            "layers.3.moe.experts_out": torch.cat([expert.down_proj.weight.T for expert in moe.experts], dim=1),
            "layers.3.moe.selection_bias": moe.gate.e_score_correction_bias,
            "layers.3.moe.shared.in": moe.shared_experts.up_proj.weight.T,
            "layers.3.moe.shared.out": moe.shared_experts.down_proj.weight.T,
        }
        result = {"ids": ids.reshape(-1).tolist(), "weights": {key: tensor(value) for key, value in weights.items()},
                  "full": tensor(full), "prefix": tensor(prefix), "prefix_state": prefix_state, "steps": steps}
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"generated {sys.argv[3]}")


if __name__ == "__main__":
    main()
