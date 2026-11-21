# Olala projected Mamba-3 reference

`reference.py` generates the small synthetic fixture consumed by the Runtime
graph test. It extracts the actual normalization, projection, prefill and
cached-step implementations from Olala's pinned upstream source and runs them
on CPU with deterministic random weights. No model checkpoint is included.

The source is
[`modeling_olala.py` at revision `b40f8165ee2254c18601220e19842600c6d91794`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/modeling_olala.py),
SHA-256 `357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf`.
The checked-in reference was generated using Python, PyTorch 2.15.0 and
einops 0.8.2:

```text
python tools/fixtures/olala-mamba3-mixer/reference.py /path/to/modeling_olala.py
```

The profile has one group, two MIMO ranks, four state dimensions, no CosNet
or post-gate normalization, and zero-centered RMSNorm. Tests compare full and
chunked prefill, three cached decode steps and all four explicit states. The
upstream prefill wraps its angle state modulo 2π; the test accounts for that
when it starts cached decoding.
