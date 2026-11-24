"""Run pinned Olala GeodesicNorm with its real layer-0 scalar parameters.

Usage: python reference.py /path/to/modeling_olala.py /path/to/layer0-geodesic.bin
Requires PyTorch 2.15.0. Both input files are verified before execution.
"""
import ast
import hashlib
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import torch

source = Path(sys.argv[1]).read_bytes()
scalars = Path(sys.argv[2]).read_bytes()
assert hashlib.sha256(source).hexdigest() == "357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf"
assert hashlib.sha256(scalars).hexdigest() == "b14815a9aad6ed3d1a74a05e4a717eb8a40c9daa24c77078d7f3ec125706d75e"
node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "OlalaGeodesicNorm")
namespace = {"torch": torch, "nn": torch.nn, "OlalaConfig": object}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "modeling_olala.py", "exec"), namespace)
mixer_bias, _, mixer_scale, mlp_bias, _, mlp_scale = struct.unpack("<6f", scalars)
x, branch = torch.tensor([[1.0, -2.0, 0.5, 4.0]]), torch.tensor([[0.7, 1.2, -1.0, 0.3]])
for name, scale, bias in [("mixer", mixer_scale, mixer_bias), ("mlp", mlp_scale, mlp_bias)]:
    module = namespace["OlalaGeodesicNorm"](SimpleNamespace(), 0).eval()
    with torch.no_grad():
        module.scale.fill_(scale)
        module.bias.fill_(bias)
        print(name, module(x, branch).tolist())
