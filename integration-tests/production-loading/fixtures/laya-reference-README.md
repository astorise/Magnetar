# Laya encoder and decision references

`laya-reference.json` compares a three-layer ModernBERT encoder connected to
the two-layer Laya decision head. The generator executes the source-verified
Transformers 5.23.0 ModernBERT embeddings, RoPE and encoder layer definitions
and the pinned Laya `DecisionModel` using PyTorch 2.15. It uses deterministic
synthetic weights and token IDs, the published Laya temperature configuration,
padding, varying option counts and all three question types. It includes raw
option and action logits, option probabilities and act probability.

The numerical test reads a synthetic Safetensors file through the bounded
Hugging Face loader and decodes its checkpoint-order tensors. A ModernBERT
Component produces the encoder graph, which the Runtime executes on the
Reference CPU using the transposed weights; its hidden states feed the Laya
reference head and both are compared with the upstream result. The graph
receives explicit test bindings: this is not the production model-loading or
decision API. No published checkpoint weights or tokenizer are involved. A
separate test checks that the published Laya encoder configuration and weight
shapes build a valid ModernBERT Component graph without loading the checkpoint.
The helper from the ModernBERT submodule verifies its source hash, and the
generator verifies the Laya source hash. With both submodules checked out:

```sh
python integration-tests/production-loading/fixtures/generate_laya_reference.py \
  path/to/modeling_modernbert.py path/to/rl_common.py
```

## Complete checkpoint comparison

`laya-real-checkpoint-reference.json` uses the complete published Laya
`model.safetensors` from revision `fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1`
(843 MB, SHA-256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`).
Its two cases have hand-specified token IDs, padding, two or three markers and
different question types. The generator checks the checkpoint and source
digests and runs the source-verified Transformers ModernBERT classes with eager
attention and the pinned Laya decision head in PyTorch float32. It records
hidden states, option/action logits and probabilities, not model weights.

The optional integration test checks the complete file digest, reads and
decodes the actual F16 weights via the bounded Hugging Face loader, constructs
the encoder Component graph, and executes it with explicit test weight bindings
on the native CPU Provider. The graph's output feeds the Laya reference head.
On the Windows test host, the two cases' maximum absolute errors were 1.15e-5
for hidden states, 2.15e-6 for option logits, 0.00147 for action logits, and
3.28e-7 for option probabilities.

`resident_laya_checkpoint_decisions_match_pytorch` separately loads the same
checkpoint through `LoadedLaya`, the authenticated Hugging Face payload source,
the registered Component and the resident Runtime encoder. Its option and
action scores and probabilities match those two PyTorch cases. It also runs
choice, score and noul questions through the pinned real tokenizer and compares
their results with independently prepared upstream token sequences. The text
fixtures include accented characters and literal mask-token text.

`laya-text-reference.json` is produced by the actual source-verified Laya
sequence builder and the pinned tokenizer (SHA-256
`6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30`).
These tests qualify the exercised f32 CPU paths, not native BF16 or GPU
execution, statistical calibration, arbitrary checkpoints or long windows.

Regenerate and run (the external checkpoint and source files are not committed):

```sh
python integration-tests/production-loading/fixtures/generate_laya_real_checkpoint_reference.py \
  /path/to/model.safetensors /path/to/modeling_modernbert.py /path/to/rl_common.py \
  integration-tests/production-loading/fixtures/laya-real-checkpoint-reference.json
MAGNETAR_LAYA_REAL_CHECKPOINT=/path/to/model.safetensors cargo test --locked \
  --manifest-path integration-tests/production-loading/Cargo.toml \
  laya_real_checkpoint_component_and_head_match_pytorch -- --ignored --nocapture
python integration-tests/production-loading/fixtures/generate_laya_text_reference.py \
  /path/to/tokenizer.json /path/to/rl_common.py \
  integration-tests/production-loading/fixtures/laya-text-reference.json
MAGNETAR_LAYA_REAL_CHECKPOINT=/path/to/model.safetensors \
MAGNETAR_LAYA_TOKENIZER=/path/to/tokenizer.json \
cargo --config 'profile.dev.package.sha2.opt-level=3' \
  --config 'profile.dev.package.magnetar-component-laya.opt-level=3' \
  --config 'profile.dev.package.magnetar-component-modernbert.opt-level=3' \
  --config 'profile.dev.package.magnetar-inference-component.opt-level=3' \
  --config 'profile.dev.package.magnetar-provider-cpu.opt-level=3' \
  test --locked --manifest-path integration-tests/production-loading/Cargo.toml \
  resident_laya_checkpoint_decisions_match_pytorch -- --ignored --nocapture
```
