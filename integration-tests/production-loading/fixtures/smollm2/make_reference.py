"""Create a tiny SmolLM2-shaped Llama bundle and independent logits.

Usage: python make_reference.py OUTPUT_DIRECTORY
Requires torch 2.15.0, transformers 5.23.0, numpy and safetensors.
The random weights are not the published SmolLM2 checkpoint.
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 100_000.0},
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=63,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config).float().eval()
    generator = torch.Generator().manual_seed(12345)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            random = torch.randn(parameter.shape, generator=generator)
            if name == "model.embed_tokens.weight":
                parameter.copy_(random)
            elif parameter.ndim == 1:
                parameter.copy_(1.0 + random * 0.1)
            else:
                parameter.copy_(random * parameter.shape[-1] ** -0.5)
    model.save_pretrained(out, safe_serialization=True)

    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [], "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
        "decoder": None, "model": {"type": "WordLevel",
            "vocab": {f"t{i}": i for i in range(64)}, "unk_token": "t0"},
    }
    for name, value in [("tokenizer.json", tokenizer), ("tokenizer_config.json", {})]:
        with open(os.path.join(out, name), "w", encoding="utf-8") as file:
            json.dump(value, file)

    prompt = [5, 17, 33, 8, 41, 2, 60, 12]
    generated, steps = [], []
    with torch.no_grad():
        for step in range(6):
            ids = torch.tensor([prompt + generated])
            logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().numpy()
            sampled = int(np.argmax(logits))
            if sampled == config.eos_token_id:
                raise RuntimeError("the reference sampled EOS before the requested decode steps")
            order = sorted(range(len(logits)), key=lambda token: (-float(logits[token]), token))[:32]
            steps.append({
                "step": step,
                "sampled_token_id": sampled,
                "vocab_size": len(logits),
                "top": [{"token_id": token, "logit": float(logits[token])} for token in order],
                "full_digest": "sha256:" + hashlib.sha256(logits.astype("<f4").tobytes()).hexdigest(),
                "non_finite_count": int(np.count_nonzero(~np.isfinite(logits))),
            })
            generated.append(sampled)
    with open(os.path.join(out, "model.safetensors"), "rb") as file:
        digest = hashlib.sha256(file.read()).hexdigest()
    reference = {
        "schema_version": "magnetar.inference-qualification.logit-reference/v1",
        "checkpoint": {"name": "tiny-smollm2-shaped-llama", "revision": None,
                       "weights_digest": "sha256:" + digest, "digest_verified": True},
        "provider": "transformers-f32", "dtype": "f32", "prompt_token_ids": prompt,
        "generated_token_ids": generated, "steps": steps,
    }
    with open(os.path.join(out, "reference.json"), "w", encoding="utf-8") as file:
        json.dump(reference, file, indent=2)
        file.write("\n")
    print("Generated tokens:", generated)


if __name__ == "__main__":
    main()
