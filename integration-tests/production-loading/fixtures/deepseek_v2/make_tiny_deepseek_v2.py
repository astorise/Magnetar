"""Tiny random DeepseekV2ForCausalLM bundles with well-scaled weights.

Variants:
  lite   no low-rank query, YaRN with mscale == mscale_all_dim (the
         published V2/V3 form), dense first layer, 4 routed experts top-2,
         2 shared experts, softmax routing, routed scaling 1.
  qlora  a low-rank query, YaRN with mscale != mscale_all_dim (cos/sin
         scaled), routed scaling 2.5.
  plain  like lite without any RoPE scaling.
Usage: make_tiny_deepseek_v2.py <out_dir> <variant>
"""
import json, os, sys
import torch
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

VOCAB = 64


def config(variant):
    rope = {"rope_type": "yarn", "rope_theta": 100.0, "factor": 4.0,
            "original_max_position_embeddings": 16, "beta_fast": 4.0, "beta_slow": 1.0,
            "mscale": 0.707, "mscale_all_dim": 0.707}
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        n_shared_experts=2, n_routed_experts=4, num_experts_per_tok=2, first_k_dense_replace=1,
        kv_lora_rank=12, q_lora_rank=None, qk_nope_head_dim=6, qk_rope_head_dim=4, v_head_dim=5,
        max_position_embeddings=256, tie_word_embeddings=False, rms_norm_eps=1e-6,
        topk_method="greedy", norm_topk_prob=False, routed_scaling_factor=1.0,
        rope_parameters=rope, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    if variant == "qlora":
        kwargs.update(q_lora_rank=8, routed_scaling_factor=2.5)
        kwargs["rope_parameters"] = dict(rope, mscale=1.0, mscale_all_dim=0.707)
    elif variant == "plain":
        kwargs["rope_parameters"] = {"rope_type": "default", "rope_theta": 100.0}
    elif variant != "lite":
        raise SystemExit(f"unknown variant {variant}")
    return DeepseekV2Config(**kwargs)


def scaled_init(model, seed):
    g = torch.Generator().manual_seed(seed)
    normal = lambda shape, std: torch.randn(shape, generator=g) * std
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == "model.embed_tokens.weight":
                p.copy_(normal(p.shape, 1.0))
            elif name == "model.norm.weight":
                p.copy_(3.0 + normal(p.shape, 0.3))
            elif p.ndim == 1:
                p.copy_(1.0 + normal(p.shape, 0.2))
            elif name.endswith("mlp.gate.weight"):
                # Decisive but not degenerate routing.
                p.copy_(normal(p.shape, 1.0))
            else:
                p.copy_(normal(p.shape, p.shape[-1] ** -0.5))


def main():
    out, variant = sys.argv[1], sys.argv[2]
    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(config(variant)).float().eval()
    scaled_init(model, {"lite": 1, "qlora": 2, "plain": 3}[variant])
    model.save_pretrained(out, safe_serialization=True)
    vocab = {f"t{i}": i for i in range(VOCAB)}
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
        "normalizer": None, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
        "decoder": None, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"},
    }
    with open(os.path.join(out, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f)
    with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
        json.dump({}, f)
    print(out, variant)


if __name__ == "__main__":
    main()
