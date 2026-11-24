"""Generate tiny mixed RoPE/NoPE SmolLM3 logits with Transformers.

The random projections come from ../smollm2/tiny; no released weights are
downloaded. Requires torch, transformers, numpy and safetensors.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import SmolLM3Config, SmolLM3ForCausalLM

ROOT = Path(__file__).parent
SOURCE = ROOT.parent / "smollm2" / "tiny"
OUT = ROOT / "tiny"
PROMPT = [5, 17, 33, 8, 41, 2, 60, 12]


def main():
    config = SmolLM3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 100_000.0},
        no_rope_layers=[1, 0, 1],
        use_sliding_window=False,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=63,
        pad_token_id=0,
    )
    model = SmolLM3ForCausalLM(config).float().eval()
    missing, unexpected = model.load_state_dict(load_file(SOURCE / "model.safetensors"), strict=False)
    if set(missing) - {"lm_head.weight"} or unexpected:
        raise RuntimeError(f"unexpected weight mismatch: missing={missing}, extra={unexpected}")
    model.tie_weights()
    OUT.mkdir(exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        shutil.copyfile(SOURCE / name, OUT / name)

    generated, steps = [], []
    with torch.no_grad():
        for step in range(6):
            ids = torch.tensor([PROMPT + generated])
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
    digest = hashlib.sha256((OUT / "model.safetensors").read_bytes()).hexdigest()
    result = {
        "schema_version": "magnetar.inference-qualification.logit-reference/v1",
        "checkpoint": {"name": "tiny-smollm3", "revision": None,
                       "weights_digest": "sha256:" + digest, "digest_verified": True},
        "provider": "transformers-f32", "dtype": "f32", "prompt_token_ids": PROMPT,
        "generated_token_ids": generated, "steps": steps,
    }
    (OUT / "reference.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("SmolLM3 generated tokens:", generated)


if __name__ == "__main__":
    main()
