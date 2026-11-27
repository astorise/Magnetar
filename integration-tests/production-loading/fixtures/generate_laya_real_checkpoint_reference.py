"""Run the pinned Laya checkpoint with independent PyTorch model classes.

Usage: python generate_laya_real_checkpoint_reference.py MODEL.safetensors \
       MODELING_MODERNBERT.py RL_COMMON.py OUTPUT.json

The output contains activations and scores for short, hand-specified token IDs,
not model weights. The generator verifies the checkpoint and source hashes.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors.torch import load_file


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


model_path, modern_path, laya_path, output = map(Path, sys.argv[1:])
assert digest(model_path) == "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
assert digest(modern_path) == "29b598c408582bd53c0a48ae34dce2cb008aefc5f6087b03c310b5ce7bb0de9b"
assert digest(laya_path) == "8d83611d480c971d640a7b7d3aa2f2219c5e8455e9cc2329fd073681bd8be23e"
root = Path(__file__).resolve().parents[3]
helper = root / "components/modernbert/tests/fixtures/generate_encoder_reference.py"
tree = ast.parse(helper.read_text(encoding="utf-8"))
cut = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
           and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets))
arguments = sys.argv[:]
sys.argv = [str(helper), str(modern_path)]
modern = {"__file__": str(helper)}
exec(compile(ast.Module(body=tree.body[:cut], type_ignores=[]), str(helper), "exec"), modern)
sys.argv = arguments
source = laya_path.read_bytes()
cls = next(node for node in ast.parse(source).body
           if isinstance(node, ast.ClassDef) and node.name == "DecisionModel")
laya = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=[cls], type_ignores=[]), "pinned_laya.py", "exec"), laya)


class Encoder(modern["Reference"]):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.last_hidden = None

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1])[None, :]
        self.last_hidden = super().forward(input_ids, attention_mask, positions, 64)
        return SimpleNamespace(last_hidden_state=self.last_hidden)


cfg_path = root / "components/modernbert/tests/fixtures/laya-encoder.config.json"
settings_path = root / "components/laya/tests/fixtures/rl_agent_config.json"
assert hashlib.sha256(cfg_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979"
assert hashlib.sha256(settings_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd"
cfg = SimpleNamespace(**json.loads(cfg_path.read_text(encoding="utf-8")))
cfg._attn_implementation = "eager"
cfg.sliding_window = 64
settings = json.loads(settings_path.read_text(encoding="utf-8"))
torch.set_num_threads(4)
model = laya["DecisionModel"](Encoder(cfg), head_layers=2, n_act=2, dropout=0.0).eval()
model.load_state_dict(load_file(str(model_path)), strict=True, assign=True)
model = model.float()
cases = [
    ([50281, 10, 50284, 20, 50284, 50282, 50283], [1, 1, 1, 1, 1, 1, 0], [2, 4], 0),
    ([50281, 50284, 17, 50284, 18, 50284, 50282], [1, 1, 1, 1, 1, 1, 1], [1, 3, 5], 1),
]
results = []
with torch.inference_mode():
    for ids, mask, markers, qtype in cases:
        logits, actions = model(torch.tensor([ids]), torch.tensor([mask]), torch.tensor([markers]),
                                torch.ones((1, len(markers)), dtype=torch.bool), torch.tensor([qtype]))
        kind = ["choice", "score", "noul"][qtype]
        bucket = "2" if len(markers) <= 2 else "3-5"
        temperature = settings["temperature_by_options"].get(f"{kind}:{bucket}", settings["temperature"][qtype])
        results.append({
            "ids": ids, "mask": mask, "markers": markers, "qtype": qtype,
            "hidden": model.encoder.last_hidden.flatten().tolist(),
            "logits": logits.flatten().tolist(), "action_logits": actions.flatten().tolist(),
            "probabilities": torch.softmax(logits.float() / temperature, -1).flatten().tolist(),
            "action_probabilities": torch.softmax(actions.float(), -1).flatten().tolist(),
        })
output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(results)} real-checkpoint reference cases to {output}")
