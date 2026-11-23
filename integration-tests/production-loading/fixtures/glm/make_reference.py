"""Generate tiny GLM-4.7-shaped checkpoints and independent Transformers logits.

Uses Transformers 5.23.0 and PyTorch 2.15.0. The fixtures intentionally use
different decoder-wide (1e-5) and latent bottleneck (1e-6) RMS epsilons.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM


ROOT = Path(__file__).parent
PROMPT = [4, 12, 7, 19]


def generate(top_k):
    torch.manual_seed(100 + top_k)
    config = Glm4MoeLiteConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, n_shared_experts=1, n_routed_experts=4,
        num_experts_per_tok=top_k, n_group=2, topk_group=1,
        kv_lora_rank=12, q_lora_rank=8, qk_nope_head_dim=6,
        qk_rope_head_dim=4, v_head_dim=5, rms_norm_eps=1e-5,
        norm_topk_prob=True, routed_scaling_factor=1.8,
        mlp_layer_types=["dense", "sparse"],
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    model = Glm4MoeLiteForCausalLM(config).float().eval()
    generator = torch.Generator().manual_seed(200 + top_k)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            random = torch.randn(parameter.shape, generator=generator)
            if name.endswith("e_score_correction_bias"):
                parameter.copy_(random * 0.2)
            elif name == "model.embed_tokens.weight":
                parameter.copy_(random)
            elif name == "model.norm.weight":
                parameter.copy_(3.0 + random * 0.3)
            elif parameter.ndim == 1:
                parameter.copy_(1.0 + random * 0.2)
            elif name.endswith("mlp.gate.weight"):
                parameter.copy_(random * 0.5)
            else:
                parameter.copy_(random * (parameter.shape[-1] ** -0.5))

    out = ROOT / f"tiny-top{top_k}"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
        "normalizer": None, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
        "decoder": None, "model": {"type": "WordLevel", "vocab": {f"t{i}": i for i in range(64)}, "unk_token": "t0"},
    }
    (out / "tokenizer.json").write_text(json.dumps(tokenizer))
    (out / "tokenizer_config.json").write_text("{}")
    generated, steps = [], []
    with torch.no_grad():
        for step in range(6):
            logits = model(input_ids=torch.tensor([PROMPT + generated]), use_cache=False).logits[0, -1].float().numpy()
            sampled = int(np.argmax(logits))
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
        "checkpoint": {
            "name": f"tiny-glm-top{top_k}", "revision": None,
            "weights_digest": "sha256:" + hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest(),
            "digest_verified": True,
        },
        "provider": "transformers",
        "dtype": "f32", "prompt_token_ids": PROMPT,
        "generated_token_ids": generated, "steps": steps,
    }
    (out / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(out, generated)


if __name__ == "__main__":
    generate(1)
    generate(2)
