"""Run Transformers' `qwen4_exp` blocks on synthetic weights and print a fixture.

Usage: python generate-qwen4-exp-reference.py > qwen4-exp-reference.json
Requires PyTorch and Transformers 5.23.0, whose `models/qwen4_exp` (Apache-2.0)
this script checks by SHA-256 before importing it. The fixture holds synthetic
weights only. See design D7 of openspec/changes/support-qwen38-flash-next.
"""

import hashlib
import json
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
json.dump(fixture, sys.stdout, indent=1)
print()
