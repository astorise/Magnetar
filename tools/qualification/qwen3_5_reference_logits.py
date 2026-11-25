"""transformers reference logits for a real Qwen3.5 checkpoint, greedy, the
whole sequence recomputed at every step (no cache), in float32, written as a
magnetar.inference-qualification.logit-reference/v1 document.

The prompt is encoded with the checkpoint's own tokenizer.json, without
special tokens, as Magnetar's tokenizer encodes it.

Usage: qwen3_5_reference_logits.py <checkpoint dir> <repo> <revision> <weights sha256>
           <n_new> <out.json> <prompt text>
Requires torch, transformers (5.23.0 was used) and tokenizers.
"""
import hashlib
import json
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import Qwen3_5ForConditionalGeneration

TOP_K = 32


def snapshot(step, logits, sampled):
    order = sorted(range(len(logits)), key=lambda t: (-float(logits[t]), t))[:TOP_K]
    return {"step": step, "sampled_token_id": sampled, "vocab_size": int(len(logits)),
            "top": [{"token_id": int(t), "logit": float(logits[t])} for t in order],
            "full_digest": "sha256:" + hashlib.sha256(logits.astype("<f4").tobytes()).hexdigest(),
            "non_finite_count": int(np.count_nonzero(~np.isfinite(logits)))}


def main():
    directory, repo, revision, weights_sha256 = sys.argv[1:5]
    n_new, out, prompt_text = int(sys.argv[5]), sys.argv[6], sys.argv[7]
    digest = hashlib.sha256()
    with open(f"{directory}/model.safetensors", "rb") as weights:
        for block in iter(lambda: weights.read(8 << 20), b""):
            digest.update(block)
    if "sha256:" + digest.hexdigest() != weights_sha256:
        raise SystemExit("model.safetensors does not match the pinned digest")
    tokenizer = Tokenizer.from_file(f"{directory}/tokenizer.json")
    prompt = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        directory, dtype=torch.float32).eval()
    generated, steps = [], []
    with torch.no_grad():
        for step in range(n_new):
            ids = torch.tensor([prompt + generated])
            logits = model(input_ids=ids).logits[0, -1].float().numpy()
            sampled = int(np.argmax(logits))
            top2 = np.sort(logits)[-2:]
            steps.append(snapshot(step, logits, sampled))
            generated.append(sampled)
            print(f"step {step}: token {sampled} logit {logits[sampled]:.4f} "
                  f"margin {top2[1] - top2[0]:.4f}", flush=True)
    corpus = {"schema_version": "magnetar.inference-qualification.logit-reference/v1",
              "checkpoint": {"name": repo, "revision": revision,
                             "weights_digest": weights_sha256, "digest_verified": True},
              "provider": "transformers-f32", "dtype": "f32", "prompt_token_ids": prompt,
              "generated_token_ids": generated, "steps": steps}
    with open(out, "w") as f:
        json.dump(corpus, f, indent=2)
        f.write("\n")
    print("prompt", prompt, "generated", generated, repr(tokenizer.decode(generated)))


if __name__ == "__main__":
    main()
