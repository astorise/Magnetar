# Kev-0.8B real-weight pointer reference

`kev-real-checkpoint-reference.json` records two independent question rows with
two and three options. The rows use actual Kev delimiter token IDs and
hand-specified content IDs. They are not tokenizer or product-API fixtures.
The generator verifies and executes all 24 Qwen3.5 backbone layers in PyTorch
float32, applying the 186 rank-16 LoRA overlays as explicit PyTorch linear
operations. It then evaluates the full trained pointer head with its saved
temperature. The recorded decision and option hidden states, scores and
probabilities contain no checkpoint weights.

Pinned source artifacts:

- Qwen/Qwen3.5-0.8B-Base, revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, `model.safetensors` SHA-256 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`.
- jaredpalmer/kev-0.8b, revision `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`, adapter SHA-256 `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a` and `head.pt` SHA-256 `f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6`.

The optional Rust test loads the full pointer weights through the bounded
Safetensors loader, builds independent encoded rows and compares the scores
and probabilities for the PyTorch-provided hidden states. It does **not** run
Qwen, activate LoRA, or reuse a prefix cache in Magnetar. The export helper
accepts only the pinned `head.pt`, uses PyTorch's restricted weights-only
loader, and writes the four tensor values to Safetensors; the test checks the
result's SHA-256 `ab7c75921e79ec27ca97d8bf7b928e277dac843a6f4a528f9e8b7ab0fe7cd47d`.
Neither checkpoint nor exported weights are checked into this repository.

Regenerate with PyTorch 2.15, Safetensors 0.8.0 and Transformers 5.23.0:

```sh
python integration-tests/production-loading/fixtures/generate_kev_real_checkpoint_reference.py \
  /path/to/qwen-bundle /path/to/adapter_model.safetensors /path/to/head.pt \
  integration-tests/production-loading/fixtures/kev-real-checkpoint-reference.json
python integration-tests/production-loading/fixtures/export_kev_pointer_head.py \
  /path/to/head.pt /path/to/kev-head.safetensors
MAGNETAR_KEV_HEAD_SAFETENSORS=/path/to/kev-head.safetensors cargo test --locked \
  --manifest-path integration-tests/production-loading/Cargo.toml \
  full_kev_head_scores_real_adapted_qwen_hidden_states -- --ignored --nocapture
```
