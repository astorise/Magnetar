"""Tiny random Qwen3_5ForCausalLM bundles with well-scaled weights.

Variants:
  hybrid  4 layers: linear, full, linear, full attention. Gated-delta mixers
          with 2 key heads and 4 value heads of 8, convolution kernel 4;
          gated attention with 2 query heads and 1 KV head of 16, a quarter
          of each head rotating; tied embeddings.
Usage: make_tiny_qwen3_5.py <out_dir> <variant>
"""
import json, os, sys
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

VOCAB = 64


def config(variant):
    if variant != "hybrid":
        raise SystemExit(f"unknown variant {variant}")
    return Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 100.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0],
                         "mrope_interleaved": True},
        max_position_embeddings=256, rms_norm_eps=1e-6, tie_word_embeddings=True,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )


def scaled_init(model, seed):
    g = torch.Generator().manual_seed(seed)
    normal = lambda shape, std: torch.randn(shape, generator=g) * std
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == "model.embed_tokens.weight":
                # Tied to the LM head: a larger scale makes one token absorb
                # every greedy step.
                p.copy_(normal(p.shape, 0.5))
            elif name.endswith("linear_attn.A_log"):
                p.copy_(torch.log(torch.empty(p.shape).uniform_(1.0, 8.0, generator=g)))
            elif name.endswith("linear_attn.dt_bias"):
                p.copy_(normal(p.shape, 0.5))
            elif name.endswith("linear_attn.conv1d.weight"):
                p.copy_(normal(p.shape, 0.5))
            elif name.endswith("linear_attn.norm.weight"):
                # The mixer's gated norm keeps its plain weight.
                p.copy_(1.0 + normal(p.shape, 0.2))
            elif p.ndim == 1:
                # Zero-centered norms: the scale is 1 + weight.
                p.copy_(normal(p.shape, 0.2))
            else:
                p.copy_(normal(p.shape, p.shape[-1] ** -0.5))


def main():
    out, variant = sys.argv[1], sys.argv[2]
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config(variant)).float().eval()
    scaled_init(model, {"hybrid": 1}[variant])
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
