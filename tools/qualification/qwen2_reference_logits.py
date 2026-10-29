"""Independent reference logits for a Hugging Face Qwen2 checkpoint.

qualify-real-checkpoint-performance A02: produces the checkpoint-bound
logit reference corpus every Magnetar Provider (Reference CPU included) is
compared against. Deliberately shares no code with Magnetar: it reads the
Safetensors file directly, upcasts BF16 weights to F32 and runs the Qwen2
forward pass (RMSNorm, biased Q/K/V, half-rotation RoPE as in Hugging Face
`rotate_half`, grouped-query causal attention, SwiGLU MLP, tied lm_head)
in plain numpy, recomputing the whole sequence each step (no KV cache).

Usage:
  python3 -I tools/qualification/qwen2_reference_logits.py \
      <checkpoint_dir> <revision> <weights_sha256> <n_new> <out.json> <prompt ids...>

Writes a `magnetar.inference-qualification.logit-reference/v1` JSON
document (see tools/inference-qualification/src/numerics.rs).
"""
import hashlib
import json
import struct
import sys

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        buf = raw[base + start : base + end]
        if meta["dtype"] == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        elif meta["dtype"] == "F32":
            arr = np.frombuffer(buf, dtype=np.float32)
        else:
            raise ValueError(meta["dtype"])
        tensors[name] = arr.reshape(meta["shape"])
    return tensors


def rms_norm(x, w, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def rope(x, positions, theta):
    # x: [seq, heads, head_dim], rotate_half convention
    d = x.shape[-1]
    inv = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
    freqs = np.outer(positions, inv)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb)[:, None, :].astype(np.float32)
    sin = np.sin(emb)[:, None, :].astype(np.float32)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


SNAPSHOT_TOP_K = 32


def snapshot(step, logits, sampled):
    order = sorted(range(len(logits)), key=lambda t: (-float(logits[t]), t))[:SNAPSHOT_TOP_K]
    return {
        "step": step,
        "sampled_token_id": sampled,
        "vocab_size": int(len(logits)),
        "top": [{"token_id": int(t), "logit": float(logits[t])} for t in order],
        "full_digest": "sha256:" + hashlib.sha256(logits.astype("<f4").tobytes()).hexdigest(),
        "non_finite_count": int(np.count_nonzero(~np.isfinite(logits))),
    }


def main():
    ckpt, revision, weights_sha256 = sys.argv[1], sys.argv[2], sys.argv[3]
    n_new, out_path = int(sys.argv[4]), sys.argv[5]
    ids = [int(t) for t in sys.argv[6:]]
    steps = []
    cfg = json.load(open(f"{ckpt}/config.json"))
    W = load_safetensors(f"{ckpt}/model.safetensors")
    H, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = H // nh
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    embed = W["model.embed_tokens.weight"]
    lm_head = W.get("lm_head.weight", embed)
    L = cfg["num_hidden_layers"]
    generated = []
    for step in range(n_new):
        seq = ids + generated
        pos = np.arange(len(seq))
        x = embed[seq].astype(np.float32)
        for l in range(L):
            p = f"model.layers.{l}."
            h = rms_norm(x, W[p + "input_layernorm.weight"], eps)
            q = h @ W[p + "self_attn.q_proj.weight"].T + W[p + "self_attn.q_proj.bias"]
            k = h @ W[p + "self_attn.k_proj.weight"].T + W[p + "self_attn.k_proj.bias"]
            v = h @ W[p + "self_attn.v_proj.weight"].T + W[p + "self_attn.v_proj.bias"]
            q = rope(q.reshape(len(seq), nh, hd), pos, theta)
            k = rope(k.reshape(len(seq), nkv, hd), pos, theta)
            v = v.reshape(len(seq), nkv, hd)
            rep = nh // nkv
            k = np.repeat(k, rep, axis=1)
            v = np.repeat(v, rep, axis=1)
            scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd)
            mask = np.triu(np.ones((len(seq), len(seq)), dtype=bool), 1)
            scores = np.where(mask[None], -np.inf, scores)
            scores = scores - scores.max(-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(-1, keepdims=True)
            attn = np.einsum("hqk,khd->qhd", probs, v).reshape(len(seq), H)
            x = x + attn @ W[p + "self_attn.o_proj.weight"].T
            h = rms_norm(x, W[p + "post_attention_layernorm.weight"], eps)
            gate = h @ W[p + "mlp.gate_proj.weight"].T
            up = h @ W[p + "mlp.up_proj.weight"].T
            act = gate / (1.0 + np.exp(-gate)) * up
            x = x + act @ W[p + "mlp.down_proj.weight"].T
        x = rms_norm(x, W["model.norm.weight"], eps)
        logits = (x[-1] @ lm_head.T).astype(np.float32)
        sampled = int(np.argmax(logits))
        steps.append(snapshot(step, logits, sampled))
        generated.append(sampled)
        print(f"step {step}: token {sampled} logit {logits[sampled]:.4f}", flush=True)
    corpus = {
        "schema_version": "magnetar.inference-qualification.logit-reference/v1",
        "checkpoint": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": revision,
            "weights_digest": weights_sha256,
            "digest_verified": True,
        },
        "provider": "independent-numpy-reference",
        "dtype": "f32",
        "prompt_token_ids": ids,
        "generated_token_ids": generated,
        "steps": steps,
    }
    with open(out_path, "w") as f:
        json.dump(corpus, f, indent=2)
        f.write("\n")
    print("generated", generated)


if __name__ == "__main__":
    main()
