"""Generate Mistral-shaped fixtures and independent NumPy reference logits.

The source weights are the checked-in randomly initialized SmolLM2-shaped
Llama fixture. Before generating sliding-window references, verify the
independent full-attention implementation against its Transformers logits.
Requires NumPy; no model weights are downloaded.
"""

import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SOURCE = ROOT.parent / "smollm2" / "tiny"
PROMPT = [5, 17, 33, 8, 41, 2, 60, 12]


def tensors(path):
    data = path.read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_size])
    payload = memoryview(data)[8 + header_size :]
    result = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise ValueError(f"unexpected dtype for {name}")
        start, end = info["data_offsets"]
        result[name] = np.frombuffer(payload[start:end], dtype="<f4").reshape(info["shape"])
    return result


def rms(x, weight):
    return x * np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-5))) * weight


def logits(weights, ids, window):
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
        z = rms(x, weights[p + "input_layernorm.weight"])
        q = rope((z @ weights[p + "self_attn.q_proj.weight"].T).reshape(rows, heads, width))
        k = rope((z @ weights[p + "self_attn.k_proj.weight"].T).reshape(rows, kv_heads, width))
        v = (z @ weights[p + "self_attn.v_proj.weight"].T).reshape(rows, kv_heads, width)
        attention = np.empty((rows, heads, width), dtype=np.float32)
        for t in range(rows):
            start = 0 if window is None else max(0, t + 1 - window)
            for head in range(heads):
                keys = k[start : t + 1, head // (heads // kv_heads)]
                values = v[start : t + 1, head // (heads // kv_heads)]
                scores = keys @ q[t, head] / np.float32(math.sqrt(width))
                probabilities = np.exp(scores - np.max(scores))
                probabilities /= np.sum(probabilities)
                attention[t, head] = probabilities @ values
        x = x + attention.reshape(rows, -1) @ weights[p + "self_attn.o_proj.weight"].T
        z = rms(x, weights[p + "post_attention_layernorm.weight"])
        gate = z @ weights[p + "mlp.gate_proj.weight"].T
        up = z @ weights[p + "mlp.up_proj.weight"].T
        x = x + ((gate / (np.float32(1.0) + np.exp(-gate))) * up) @ weights[p + "mlp.down_proj.weight"].T
    return (rms(x, weights["model.norm.weight"]) @ weights["model.embed_tokens.weight"].T)[-1]


def main():
    weights = tensors(SOURCE / "model.safetensors")
    source_reference = json.loads((SOURCE / "reference.json").read_text())
    generated = []
    for step in source_reference["steps"]:
        result = logits(weights, PROMPT + generated, None)
        for item in step["top"]:
            if abs(float(result[item["token_id"]]) - item["logit"]) > 5e-4:
                raise AssertionError("NumPy full attention disagrees with the Transformers reference")
        generated.append(step["sampled_token_id"])

    for name, window in [("full", None), ("window3", 3)]:
        out = ROOT / name
        out.mkdir(exist_ok=True)
        for file in ["model.safetensors", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
            shutil.copyfile(SOURCE / file, out / file)
        config = json.loads((SOURCE / "config.json").read_text())
        config.update({"architectures": ["MistralForCausalLM"], "model_type": "mistral", "sliding_window": window})
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        if window is None:
            shutil.copyfile(SOURCE / "reference.json", out / "reference.json")
            continue
        generated, steps = [], []
        for index in range(6):
            result = logits(weights, PROMPT + generated, window)
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
        reference = {"schema_version": "magnetar.inference-qualification.logit-reference/v1",
                     "checkpoint": {"name": "tiny-mistral-window3", "revision": None,
                                    "weights_digest": "sha256:" + digest, "digest_verified": True},
                     "provider": "numpy-f32", "dtype": "f32", "prompt_token_ids": PROMPT,
                     "generated_token_ids": generated, "steps": steps}
        (out / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
        print(name, generated)


if __name__ == "__main__":
    main()
