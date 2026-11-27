# Synthetic Laya encoder and decision reference

`laya-reference.json` compares a three-layer ModernBERT encoder connected to
the two-layer Laya decision head. The generator executes the source-verified
Transformers 5.23.0 ModernBERT embeddings, RoPE and encoder layer definitions
and the pinned Laya `DecisionModel` using PyTorch 2.15. It uses deterministic
synthetic weights and token IDs, the published Laya temperature configuration,
padding, varying option counts and all three question types. It includes raw
option and action logits, option probabilities and act probability.

The numerical test reads a synthetic Safetensors file through the bounded
Hugging Face loader, decodes its checkpoint-order tensors, and builds its
encoder and head from them. No published checkpoint weights, tokenizer or
Magnetar execution graph are involved in that numerical comparison. A separate
test checks that the published Laya encoder configuration and derived weight
shapes build a valid ModernBERT Component graph; it does not execute the graph
or load the published checkpoint.
The helper from the ModernBERT submodule verifies its source hash, and the
generator verifies the Laya source hash. With both submodules checked out:

```sh
python integration-tests/production-loading/fixtures/generate_laya_reference.py \
  path/to/modeling_modernbert.py path/to/rl_common.py
```
