"""Create a tiny OLMo 3-shaped checkpoint and independent NumPy logits.

Uses generated projection weights from ../smollm2/tiny and creates the
additional learned normalization tensors locally. No checkpoint is fetched.
Requires NumPy. The computation follows the public Olmo3 decoder equations.
"""

import hashlib
import importlib.util
import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SOURCE = ROOT.parent / "smollm2" / "tiny"
spec = importlib.util.spec_from_file_location("mistral_reference", ROOT.parent / "mistral" / "make_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def save_tensors(path, weights):
    header, payload = {}, bytearray()
    for name in sorted(weights):
        value = np.asarray(weights[name], dtype="<f4")
        start = len(payload)
        payload.extend(value.tobytes())
        header[name] = {"dtype": "F32", "shape": list(value.shape), "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def logits(weights, ids):
    x = weights["model.embed_tokens.weight"][ids].copy()
    rows, heads, kv_heads, width = len(ids), 4, 2, 8
    inv = np.float32(1.0) / np.power(np.float32(100000.0), np.arange(0, width, 2, dtype=np.float32) / width)
    angle = np.outer(np.arange(rows, dtype=np.float32), inv)
    cos = np.concatenate((np.cos(angle), np.cos(angle)), axis=-1)[:, None, :]
    sin = np.concatenate((np.sin(angle), np.sin(angle)), axis=-1)[:, None, :]

    def rope(value):
        rotated = np.concatenate((-value[..., width // 2 :], value[..., : width // 2]), axis=-1)
        return value * cos + rotated * sin

    for layer in range(3):
        p = f"model.layers.{layer}."
        q = reference.rms(x @ weights[p + "self_attn.q_proj.weight"].T, weights[p + "self_attn.q_norm.weight"])
        k = reference.rms(x @ weights[p + "self_attn.k_proj.weight"].T, weights[p + "self_attn.k_norm.weight"])
        q = rope(q.reshape(rows, heads, width))
        k = rope(k.reshape(rows, kv_heads, width))
        v = (x @ weights[p + "self_attn.v_proj.weight"].T).reshape(rows, kv_heads, width)
        attention = np.empty((rows, heads, width), dtype=np.float32)
        for t in range(rows):
            for head in range(heads):
                keys = k[: t + 1, head // (heads // kv_heads)]
                values = v[: t + 1, head // (heads // kv_heads)]
                scores = keys @ q[t, head] / np.float32(math.sqrt(width))
                probabilities = np.exp(scores - np.max(scores))
                probabilities /= np.sum(probabilities)
                attention[t, head] = probabilities @ values
        projected = attention.reshape(rows, -1) @ weights[p + "self_attn.o_proj.weight"].T
        x = x + reference.rms(projected, weights[p + "post_attention_layernorm.weight"])
        gate = x @ weights[p + "mlp.gate_proj.weight"].T
        up = x @ weights[p + "mlp.up_proj.weight"].T
        down = ((gate / (np.float32(1.0) + np.exp(-gate))) * up) @ weights[p + "mlp.down_proj.weight"].T
        x = x + reference.rms(down, weights[p + "post_feedforward_layernorm.weight"])
    return (reference.rms(x, weights["model.norm.weight"]) @ weights["model.embed_tokens.weight"].T)[-1]


def main():
    weights = reference.tensors(SOURCE / "model.safetensors")
    rng = np.random.default_rng(304)
    for layer in range(3):
        p = f"model.layers.{layer}."
        del weights[p + "input_layernorm.weight"]
        for name, width in [("self_attn.q_norm.weight", 32), ("self_attn.k_norm.weight", 16),
                            ("post_feedforward_layernorm.weight", 32)]:
            weights[p + name] = (1.0 + rng.normal(0.0, 0.1, width)).astype(np.float32)
    out = ROOT / "tiny"
    out.mkdir(exist_ok=True)
    for name in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        shutil.copyfile(SOURCE / name, out / name)
    save_tensors(out / "model.safetensors", weights)
    config = json.loads((SOURCE / "config.json").read_text())
    config.update({"architectures": ["Olmo3ForCausalLM"], "model_type": "olmo3",
                   "layer_types": ["full_attention"] * 3, "sliding_window": None})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    generated, steps = [], []
    for index in range(6):
        result = logits(weights, reference.PROMPT + generated)
        sampled = int(np.argmax(result))
        if sampled == 63:
            raise RuntimeError("unexpected early EOS")
        order = sorted(range(len(result)), key=lambda token: (-float(result[token]), token))[:32]
        steps.append({"step": index, "sampled_token_id": sampled, "vocab_size": len(result),
                      "top": [{"token_id": token, "logit": float(result[token])} for token in order],
                      "full_digest": "sha256:" + hashlib.sha256(result.astype("<f4").tobytes()).hexdigest(),
                      "non_finite_count": int(np.count_nonzero(~np.isfinite(result)))})
        generated.append(sampled)
    digest = hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest()
    result = {"schema_version": "magnetar.inference-qualification.logit-reference/v1",
              "checkpoint": {"name": "tiny-olmo3", "revision": None,
                             "weights_digest": "sha256:" + digest, "digest_verified": True},
              "provider": "numpy-f32", "dtype": "f32", "prompt_token_ids": reference.PROMPT,
              "generated_token_ids": generated, "steps": steps}
    (out / "reference.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("OLMo 3 tokens:", generated)


if __name__ == "__main__":
    main()
