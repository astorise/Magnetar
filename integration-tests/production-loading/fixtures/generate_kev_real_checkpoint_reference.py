"""Generate Kev Qwen3.5 + real LoRA + pointer-head PyTorch references.

Usage: python generate_kev_real_checkpoint_reference.py QWEN_DIR ADAPTER HEAD_PT OUT.json
Requires torch, safetensors and Transformers 5.23.0. The Qwen directory holds
the pinned config and the base weights as model.safetensors. Hand-specified
token IDs exercise two independent question rows; no model weights are saved.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
import transformers
from safetensors.torch import load_file
from transformers import Qwen3_5ForConditionalGeneration


def checked(path, expected):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha.update(block)
    if sha.hexdigest() != expected:
        raise ValueError(f"unexpected digest: {path}")


directory, adapter_path, head_path, output_path = map(Path, sys.argv[1:])
assert transformers.__version__ == "5.23.0"
checked(directory / "model.safetensors", "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c")
checked(adapter_path, "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a")
checked(head_path, "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6")
config = (directory / "config.json").read_bytes().replace(b"\r\n", b"\n")
assert hashlib.sha256(config).hexdigest() == "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
torch.set_num_threads(4)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    str(directory), dtype=torch.float32, attn_implementation="eager", local_files_only=True
).eval()
backbone = model.model.language_model
factors = load_file(str(adapter_path))
assert len(factors) == 372
handles = []
for name, a in factors.items():
    if not name.endswith(".lora_A.weight"):
        continue
    prefix = name.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
    b = factors[f"base_model.model.{prefix}.lora_B.weight"]
    layer = backbone.get_submodule(prefix)
    assert tuple(a.shape) == (16, layer.in_features)
    assert tuple(b.shape) == (layer.out_features, 16)
    def apply_adapter(_module, inputs, result, a=a, b=b):
        return result + 2.0 * functional.linear(functional.linear(inputs[0], a), b)
    handles.append(layer.register_forward_hook(apply_adapter))
assert len(handles) == 186
checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
head, temperature = checkpoint["head"], checkpoint["temperature"]
assert checkpoint["base_revision"] == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
assert not checkpoint["option_isolation"] and not checkpoint["special_embeddings"]
prefix = [248060, 271]
rows = [
    ([248061, 10, 248049, 20, 248050, 248049, 21, 248050, 248062], [4, 7]),
    ([248061, 11, 248049, 22, 248050, 248049, 23, 248050, 248049, 24, 248050, 248062], [4, 7, 10]),
]
cases = []
with torch.inference_mode():
    for row, options in rows:
        ids = prefix + row
        hidden = backbone(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0]
        decision = hidden[-1]
        choices = hidden[torch.tensor([len(prefix) + index for index in options])]
        query = functional.linear(decision, head["q.weight"], head["q.bias"])
        keys = functional.linear(choices, head["k.weight"], head["k.bias"])
        logits = (keys @ query) / math.sqrt(256) / temperature
        vocabulary_logits = model.lm_head(decision).float()
        probe_ids = [0, 1, 10, 20, 42, 100, 1000, 248049, 248060, 248062]
        top = torch.topk(vocabulary_logits, 16).indices.tolist()
        probe_ids = list(dict.fromkeys(probe_ids + top))
        cases.append({"prefix": prefix, "row": row, "option_end_indices": options,
                      "decision_index": len(row) - 1, "decision": decision.tolist(),
                      "options": choices.tolist(), "logits": logits.tolist(),
                      "probabilities": torch.softmax(logits, -1).tolist(),
                      "vocab_probe_ids": probe_ids,
                      "vocab_probe_logits": vocabulary_logits[probe_ids].tolist(),
                      "vocab_argmax": int(vocabulary_logits.argmax())})
output_path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(cases)} adapted Kev question references to {output_path}")
