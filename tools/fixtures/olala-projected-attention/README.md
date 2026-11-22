# Olala projected attention reference

`reference.py` runs the complete pinned upstream Olala differential
tensor-product attention class with deterministic synthetic weights. It
records the attention output, the two previous **raw** values and the attention
histories (normalized keys and shifted values) for full and chunked prefill and cached steps.
The tiny fixture follows the published no-RoPE, biasless, zero-centered
Q/K-RMSNorm, token-shift and scalable-softmax profile, with unequal signal and
noise head counts, a left window and score softcap.

The generator reuses the source extraction and cache helpers in
`../olala-differential-attention/reference.py`. They enforce SHA-256
`357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf`
for [`modeling_olala.py` at revision `b40f8165ee2254c18601220e19842600c6d91794`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/modeling_olala.py).
The fixture was generated with PyTorch 2.15.0:

```text
python tools/fixtures/olala-projected-attention/reference.py /path/to/modeling_olala.py
```

The explicit histories exercised here do not implement Runtime session state,
Component construction or end-to-end checkpoint loading.

`real_checkpoint_reference.py` runs the first checkpoint attention layer
(layer 4) with actual weights converted to f32 and synthetic hidden states.
It checks the source, header and two byte-range hashes before writing binary
reference outputs. Ranges relative to the Safetensors data section are:

- `[55829544, 55829736)` as `olala-attention-layer4-f32.bin`.
- `[11481542688, 11503968800)` as `olala-attention-layer4-bf16.bin`.

With the full header as `olala-header.bin` and the source as
`modeling-olala.py`, run from the repository root:

```text
python tools/fixtures/olala-projected-attention/real_checkpoint_reference.py target/modeling-olala.py target/olala-header.bin target/olala-attention-layer4-f32.bin target/olala-attention-layer4-bf16.bin target/olala-attention-reference
```

Set `MAGNETAR_OLALA_ATTENTION_DIR` to the absolute path of `target`, then run
`cargo test --manifest-path submodule-workspace/Cargo.toml -p magnetar-integration-production-loading --lib real_olala_attention_weights_match_pytorch_in_f32 -- --ignored --nocapture`.
The test compares the output, both raw token-shift states and both attention
histories for full prefill, a prefix and two cached steps. The upstream eager
attention casts Q/K/V and its result to bfloat16 at the attention boundary;
the surrounding projections here run in f32. This is not full-model or native
BF16 inference.
