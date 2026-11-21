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
