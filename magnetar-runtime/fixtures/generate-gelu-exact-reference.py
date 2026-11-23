"""Generate f32 GELU references using PyTorch 2.15's erf-defined path."""
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

assert torch.__version__.startswith("2.15.")
values = torch.cat((torch.linspace(-12, 12, 193), torch.tensor([-1e-6, -0.01, 0.01, 1e-6])))
expected = functional.gelu(values, approximate="none")
Path(__file__).with_name("gelu-exact-reference.json").write_text(
    json.dumps({"input": values.tolist(), "output": expected.tolist()}, indent=2) + "\n", encoding="utf-8"
)
