"""Create synthetic mixed-FP8 DeepSeek V2 weights and Transformers logits.

Requires torch 2.15.0, transformers 5.23.0, numpy and safetensors. One latent
projection is split after conversion and one routed expert is stacked after
conversion. These are random test weights, not a released checkpoint.
"""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

root = Path(__file__).resolve().parent
base = root.parent / "deepseek_v2" / "plain"
out = root / "deepseek"
out.mkdir(exist_ok=True)
for filename in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    if (base / filename).exists():
        shutil.copyfile(base / filename, out / filename)
config = json.loads((base / "config.json").read_text(encoding="utf-8"))
model_config = DeepseekV2Config.from_dict(config)
weights = load_file(base / "model.safetensors")
reference_weights = dict(weights)
block = 7
for name in (
    "model.layers.0.self_attn.kv_b_proj.weight",
    "model.layers.1.mlp.experts.0.gate_proj.weight",
):
    original = weights[name]
    rows, columns = original.shape
    scales = torch.empty((int(np.ceil(rows / block)), int(np.ceil(columns / block))), dtype=torch.float32)
    quantized = torch.empty_like(original, dtype=torch.float8_e4m3fn)
    for block_row in range(scales.shape[0]):
        for block_column in range(scales.shape[1]):
            row = slice(block_row * block, min((block_row + 1) * block, rows))
            column = slice(block_column * block, min((block_column + 1) * block, columns))
            values = original[row, column]
            scale = max(float(values.abs().max()) / 448.0, 1e-12)
            scales[block_row, block_column] = scale
            quantized[row, column] = (values / scale).to(torch.float8_e4m3fn)
    expanded = scales.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:rows, :columns]
    reference_weights[name] = quantized.float() * expanded
    weights[name] = quantized
    weights[name + "_scale_inv"] = scales
save_file(weights, out / "model.safetensors")
config["quantization_config"] = {
    "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
    "weight_block_size": [block, block],
}
(out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
model = DeepseekV2ForCausalLM(model_config).float().eval()
# Transformers 5 stores routed experts fused in memory while its checkpoint
# converter accepts the individual tensors saved in the fixture.
for layer in range(model_config.num_hidden_layers):
    prefix = f"model.layers.{layer}.mlp.experts"
    if f"{prefix}.0.gate_proj.weight" not in reference_weights:
        continue
    gate_up, down = [], []
    for expert in range(model_config.num_experts):
        gate = reference_weights.pop(f"{prefix}.{expert}.gate_proj.weight")
        up = reference_weights.pop(f"{prefix}.{expert}.up_proj.weight")
        gate_up.append(torch.cat((gate, up), dim=0))
        down.append(reference_weights.pop(f"{prefix}.{expert}.down_proj.weight"))
    reference_weights[f"{prefix}.gate_up_proj"] = torch.stack(gate_up)
    reference_weights[f"{prefix}.down_proj"] = torch.stack(down)
missing, unexpected = model.load_state_dict(reference_weights, strict=False)
if missing or unexpected:
    raise RuntimeError((missing, unexpected))
prompt = json.loads((base / "reference.json").read_text(encoding="utf-8"))["prompt_token_ids"]
generated, steps = [], []
with torch.no_grad():
    for step in range(6):
        logits = model(input_ids=torch.tensor([prompt + generated]), use_cache=False).logits[0, -1].float().numpy()
        sampled = int(np.argmax(logits))
        if sampled == model_config.eos_token_id:
            raise RuntimeError("reference sampled EOS before all decode steps")
        order = sorted(range(len(logits)), key=lambda token: (-float(logits[token]), token))[:32]
        steps.append({
            "step": step, "sampled_token_id": sampled, "vocab_size": len(logits),
            "top": [{"token_id": token, "logit": float(logits[token])} for token in order],
            "full_digest": "sha256:" + hashlib.sha256(logits.astype("<f4").tobytes()).hexdigest(),
            "non_finite_count": int(np.count_nonzero(~np.isfinite(logits))),
        })
        generated.append(sampled)
reference = {
    "schema_version": "magnetar.inference-qualification.logit-reference/v1",
    "checkpoint": {"name": "tiny-deepseek-v2-fp8-e4m3fn", "revision": None,
                   "weights_digest": "sha256:" + hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest(),
                   "digest_verified": True},
    "provider": "transformers-f32", "dtype": "f32", "prompt_token_ids": prompt,
    "generated_token_ids": generated, "steps": steps,
}
(out / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
print("Generated tokens:", generated)
