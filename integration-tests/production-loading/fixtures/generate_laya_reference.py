"""Generate a synthetic encoder-to-decision reference using upstream classes.

Usage: python generate_laya_reference.py MODELING_MODERNBERT.py RL_COMMON.py
The ModernBERT layer helper verifies the first source; the Laya class digest
is verified below. Both use synthetic deterministic weights and inputs.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

root = Path(__file__).resolve().parents[3]
helper = root / "components/modernbert/tests/fixtures/generate_encoder_reference.py"
tree = ast.parse(helper.read_text(encoding="utf-8"))
cut = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
           and any(isinstance(target, ast.Name) and target.id == "results" for target in node.targets))
arguments = sys.argv[:]
sys.argv = [str(helper), arguments[1]]
modern = {"__file__": str(helper)}
exec(compile(ast.Module(body=tree.body[:cut], type_ignores=[]), str(helper), "exec"), modern)
sys.argv = arguments

source = Path(arguments[2]).read_bytes()
assert hashlib.sha256(source).hexdigest() == "8d83611d480c971d640a7b7d3aa2f2219c5e8455e9cc2329fd073681bd8be23e"
cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "DecisionModel")
laya = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=[cls], type_ignores=[]), "pinned_laya.py", "exec"), laya)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.encoder = modern["Reference"](cfg)

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1])[None, :]
        return SimpleNamespace(last_hidden_state=self.encoder(input_ids, attention_mask, positions, 2))


cfg = SimpleNamespace(vocab_size=24, hidden_size=8, intermediate_size=12, num_attention_heads=2,
    pad_token_id=0, norm_eps=1e-5, norm_bias=False, embedding_dropout=0.0,
    mlp_bias=False, mlp_dropout=0.0, hidden_activation="gelu", attention_bias=False,
    attention_dropout=0.0, deterministic_flash_attn=False, sliding_window=2,
    layer_types=["full_attention", "sliding_attention", "full_attention"],
    max_position_embeddings=64, _attn_implementation="eager",
    rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 160000.0},
                     "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}})
model = laya["DecisionModel"](Encoder(cfg), head_layers=2, n_act=2, dropout=0.0).eval()
with torch.no_grad():
    for name, parameter in model.named_parameters():
        parameter.copy_(modern["patterned"](name, parameter.shape))

settings = json.loads((root / "components/laya/tests/fixtures/rl_agent_config.json").read_text(encoding="utf-8"))
cases = [
    ([1, 4, 6, 3, 9, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 3, 4], 0),
    ([7, 5, 3, 2, 1], [1, 1, 1, 1, 1], [1, 3], 1),
    ([9, 8, 2, 4, 0], [1, 1, 1, 1, 0], [1, 2], 2),
]
results = []
with torch.no_grad():
    for ids, mask, markers, qtype in cases:
        logits, action = model(torch.tensor([ids]), torch.tensor([mask]), torch.tensor([markers]),
                               torch.ones((1, len(markers)), dtype=torch.bool), torch.tensor([qtype]))
        kind = ["choice", "score", "noul"][qtype]
        bucket = "2" if len(markers) <= 2 else "3-5"
        temperature = settings["temperature_by_options"].get(f"{kind}:{bucket}", settings["temperature"][qtype])
        results.append({"ids": ids, "mask": mask, "markers": markers, "qtype": qtype,
                        "logits": logits.flatten().tolist(), "action_logits": action.flatten().tolist(),
                        "probabilities": torch.softmax(logits.float() / temperature, -1).flatten().tolist(),
                        "action_probability": torch.softmax(action.float(), -1)[0, 0].item()})
Path(__file__).with_name("laya-reference.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
