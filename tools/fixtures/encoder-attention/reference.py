"""Generate bidirectional attention cases from pinned Transformers sources.

Usage: python reference.py MODELING_MODERNBERT.py MASKING_UTILS.py
Requires PyTorch 2.15. Both upstream source files are SHA-256 checked before
their actual attention and mask definitions run. Inputs are synthetic.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

MODEL_HASH = "29b598c408582bd53c0a48ae34dce2cb008aefc5f6087b03c310b5ce7bb0de9b"
MASK_HASH = "fc05cdaf4cc8c93ab5cded94cea88ee79b465984915a43057093ea6b958b6268"


def load(path, digest, names, namespace):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise ValueError(f"unexpected source hash: {path}")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError("missing upstream function")
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def main():
    namespace = {"torch": torch, "nn": nn}
    load(sys.argv[1], MODEL_HASH, {"eager_attention_forward"}, namespace)
    load(sys.argv[2], MASK_HASH, {"sliding_window_bidirectional_overlay", "padding_mask_function"}, namespace)
    torch.manual_seed(29754)
    tokens, heads, dim = 6, 2, 2
    q, k, v = [torch.randn(tokens, heads, dim) * 0.8 for _ in range(3)]
    cases = []
    for name, valid, radius, scale in [
        ("global", [1] * tokens, None, None),
        ("padded_global", [1, 1, 1, 1, 0, 0], None, None),
        ("padded_local", [1, 1, 1, 1, 0, 0], 2, None),
        ("self_only", [1] * tokens, 0, None),
        ("custom_scale", [1, 1, 1, 1, 0, 0], 2, 0.7),
    ]:
        valid = torch.tensor([valid], dtype=torch.bool)
        padding = namespace["padding_mask_function"](valid)
        window = namespace["sliding_window_bidirectional_overlay"](radius) if radius is not None else None
        allowed = torch.tensor([[bool(padding(0, 0, i, j)) and (window is None or window(0, 0, i, j))
                                 for j in range(tokens)] for i in range(tokens)])
        if not allowed.any(dim=-1).all():
            raise ValueError("fixture contains a query with no allowed key")
        mask = torch.zeros(1, 1, tokens, tokens).masked_fill(~allowed, torch.finfo(torch.float32).min)
        def transform(x):
            return x.permute(1, 0, 2).unsqueeze(0)
        with torch.no_grad():
            result, _ = namespace["eager_attention_forward"](
                SimpleNamespace(training=False), transform(q), transform(k), transform(v), mask,
                scaling=scale if scale is not None else dim ** -0.5, dropout=0.0,
            )
        cases.append({"name": name, "mask": valid.reshape(-1).float().tolist(), "radius": radius,
                      "scale": scale, "output": result.reshape(-1).tolist()})
    result = {"tokens": tokens, "heads": heads, "head_dim": dim,
              "q": q.reshape(-1).tolist(), "k": k.reshape(-1).tolist(), "v": v.reshape(-1).tolist(),
              "cases": cases}
    destination = Path(__file__).resolve().parents[3] / "magnetar-runtime/fixtures/encoder-attention-reference.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
