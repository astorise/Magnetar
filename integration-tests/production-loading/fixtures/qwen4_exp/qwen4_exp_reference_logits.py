"""transformers reference logits for a tiny Qwen4Exp bundle, greedy, the
whole sequence recomputed at every step (no KV cache), written as a
magnetar.inference-qualification.logit-reference/v1 document.
Usage: qwen4_exp_reference_logits.py <bundle> <n_new> <out.json> <prompt ids...>"""
import hashlib, json, sys
import numpy as np
import torch
from transformers import Qwen4ExpForCausalLM

TOP_K = 32


def snapshot(step, logits, sampled):
    order = sorted(range(len(logits)), key=lambda t: (-float(logits[t]), t))[:TOP_K]
    return {"step": step, "sampled_token_id": sampled, "vocab_size": int(len(logits)),
            "top": [{"token_id": int(t), "logit": float(logits[t])} for t in order],
            "full_digest": "sha256:" + hashlib.sha256(logits.astype("<f4").tobytes()).hexdigest(),
            "non_finite_count": int(np.count_nonzero(~np.isfinite(logits)))}


def main():
    bundle, n_new, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    prompt = [int(t) for t in sys.argv[4:]]
    model = Qwen4ExpForCausalLM.from_pretrained(bundle, dtype=torch.float32).eval()
    generated, steps = [], []
    with torch.no_grad():
        for step in range(n_new):
            ids = torch.tensor([prompt + generated])
            logits = model(input_ids=ids).logits[0, -1].float().numpy()
            sampled = int(np.argmax(logits))
            top2 = np.sort(logits)[-2:]
            steps.append(snapshot(step, logits, sampled))
            generated.append(sampled)
            print(f"step {step}: token {sampled} logit {logits[sampled]:.4f} margin {top2[1]-top2[0]:.4f}")
    digest = hashlib.sha256(open(f"{bundle}/model.safetensors", "rb").read()).hexdigest()
    corpus = {"schema_version": "magnetar.inference-qualification.logit-reference/v1",
              "checkpoint": {"name": "tiny-qwen4-exp-bundle", "revision": None,
                             "weights_digest": "sha256:" + digest, "digest_verified": True},
              "provider": "transformers-f32", "dtype": "f32", "prompt_token_ids": prompt,
              "generated_token_ids": generated, "steps": steps}
    with open(out, "w") as f:
        json.dump(corpus, f, indent=2)
        f.write("\n")
    print("generated", generated)


if __name__ == "__main__":
    main()
