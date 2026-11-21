"""Create tiny mixed-FP8 Kimi checkpoints and logits from pinned upstream code.

Run with PyTorch 2.15.0 and Transformers 4.48.3. Pass the parent of a
`kimi_reference` package containing the two original files listed below.
The script verifies their hashes before importing them.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import save_file


source_root = Path(sys.argv[1]).resolve()
source = source_root / "kimi_reference"
for filename, digest in {
    "modeling_deepseek.py": "ce753fc9076ce13ce9557e680e2f5bd68ae8929a8225772968a9a22a1d17b7d4",
    "configuration_deepseek.py": "54f29dd0d56e9a13f639c703e524daa71794e360050012aea3788ad7cfc38b17",
}.items():
    assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == digest
sys.path.insert(0, str(source_root))
from kimi_reference.configuration_deepseek import DeepseekV3Config
from kimi_reference.modeling_deepseek import DeepseekV3ForCausalLM

ROOT = Path(__file__).parent
PROMPT = [4, 12, 7, 19]


def generate(top_k):
    torch.manual_seed(300 + top_k)
    config = DeepseekV3Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_shared_experts=1, n_routed_experts=4, num_experts_per_tok=top_k,
        first_k_dense_replace=1, moe_layer_freq=1, n_group=1, topk_group=1,
        kv_lora_rank=12, q_lora_rank=8, qk_nope_head_dim=6, qk_rope_head_dim=4,
        v_head_dim=5, max_position_embeddings=256, tie_word_embeddings=False,
        rms_norm_eps=1e-6, topk_method="noaux_tc", scoring_func="sigmoid",
        norm_topk_prob=True, routed_scaling_factor=2.827, rope_theta=100.0,
        rope_scaling={"type": "yarn", "factor": 4.0, "beta_fast": 1.0,
                      "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0,
                      "original_max_position_embeddings": 16},
        bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False,
        num_nextn_predict_layers=0,
    )
    model = DeepseekV3ForCausalLM(config).float().eval()
    generator = torch.Generator().manual_seed(400 + top_k)
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

    tensors = {name: tensor.detach().clone().contiguous() for name, tensor in model.state_dict().items()}
    quantized_names = [
        "model.layers.1.self_attn.q_b_proj.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
    ]
    for name in quantized_names:
        weight = tensors[name]
        scale = float(weight.abs().max()) / 448.0
        encoded = (weight / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        with torch.no_grad():
            model.get_parameter(name).copy_(encoded.float() * scale)
        tensors[name] = encoded.contiguous()
        tensors[name + "_scale_inv"] = torch.tensor([[scale]], dtype=torch.float32)
    # The real Kimi shards contain these stale, nonpersistent BF16 buffers.
    for layer in range(config.num_hidden_layers):
        tensors[f"model.layers.{layer}.self_attn.rotary_emb.inv_freq"] = torch.zeros(56, dtype=torch.bfloat16)

    out = ROOT / f"tiny-top{top_k}"
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "model.safetensors")
    config_json = config.to_dict()
    config_json["model_type"] = "kimi_k2"
    config_json["architectures"] = ["DeepseekV3ForCausalLM"]
    config_json["quantization_config"] = {
        "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    (out / "config.json").write_text(json.dumps(config_json, indent=2) + "\n")
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
        "checkpoint": {"name": f"tiny-kimi-k2-fp8-top{top_k}", "revision": None,
                       "weights_digest": "sha256:" + hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest(),
                       "digest_verified": True},
        "provider": "pinned-kimi-transformers", "dtype": "f32", "prompt_token_ids": PROMPT,
        "generated_token_ids": generated, "steps": steps,
    }
    (out / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(out, generated)


if __name__ == "__main__":
    generate(1)
    generate(2)
