"""Rewrites a tiny Qwen4Exp bundle's routed experts, stored one by one
(`mlp.experts.E.{gate,up,down}_proj.weight`), in the stacked layout that
`transformers` holds in memory (and that other checkpoints store):
`mlp.experts.gate_up_proj` `[E, 2 * I, H]` (each expert's gate rows, then its
up rows) and `mlp.experts.down_proj` `[E, H, I]`. Every other file and
tensor is copied unchanged, so the bundle computes the same logits.
Usage: stack_qwen4_exp_experts.py <bundle> <out_dir>"""
import os, re, shutil, sys
import torch
from safetensors.torch import load_file, save_file

EXPERT = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$")


def main():
    bundle, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    tensors = load_file(os.path.join(bundle, "model.safetensors"))
    experts, kept = {}, {}
    for name, value in tensors.items():
        match = EXPERT.match(name)
        if match:
            experts.setdefault(match.group(1), {}).setdefault(int(match.group(2)), {})[match.group(3)] = value
        else:
            kept[name] = value
    for prefix, by_expert in experts.items():
        count = len(by_expert)
        assert sorted(by_expert) == list(range(count)), prefix
        kept[f"{prefix}.gate_up_proj"] = torch.stack(
            [torch.cat([by_expert[e]["gate"], by_expert[e]["up"]], dim=0) for e in range(count)]).contiguous()
        kept[f"{prefix}.down_proj"] = torch.stack([by_expert[e]["down"] for e in range(count)]).contiguous()
    save_file(kept, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for name in os.listdir(bundle):
        if name not in ("model.safetensors", "reference.json"):
            shutil.copy(os.path.join(bundle, name), os.path.join(out, name))
    print(out, len(experts), "layers of stacked experts")


if __name__ == "__main__":
    main()
