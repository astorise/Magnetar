"""Export the verified, pinned Kev pointer head to Safetensors.

Usage: python export_kev_pointer_head.py HEAD.pt OUT.safetensors
Requires PyTorch with its restricted weights-only loader and safetensors.
This helper accepts exactly the published checkpoint identified below.
"""
import hashlib
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

source, output = map(Path, sys.argv[1:])
if hashlib.sha256(source.read_bytes()).hexdigest() != "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6":
    raise ValueError("unexpected Kev pointer checkpoint digest")
checkpoint = torch.load(source, map_location="cpu", weights_only=True)
head = checkpoint["head"]
assert set(head) == {"q.weight", "q.bias", "k.weight", "k.bias"}
assert checkpoint["temperature"] == 2.3510958125672174
for name, tensor in head.items():
    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == ((256, 1024) if name.endswith(".weight") else (256,))
save_file({name: tensor.contiguous() for name, tensor in head.items()}, str(output))
print("Exported SHA-256:", hashlib.sha256(output.read_bytes()).hexdigest())
