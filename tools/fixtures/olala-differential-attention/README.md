# Olala differential attention reference

`reference.py` executes the pinned upstream attention class using deterministic
synthetic weights and captures the actual Q/K/V tensors at the eager attention
boundary. The fixture records the class's outputs for a full prefill, two
cached multi-token chunks and a single-token step. It exercises six signal
heads and two noise heads, nonzero token shifts, Q/K normalization, scalable
softmax, a left window and score softcap. The source casts Q/K/V and attention
outputs to bfloat16; the graph operator's explicit rounding profile models
those conversions while accumulating in f32.

Source: [`modeling_olala.py` at revision `b40f8165ee2254c18601220e19842600c6d91794`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/modeling_olala.py),
SHA-256 `357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf`.
The checked-in fixture was generated with PyTorch 2.15.0:

```text
python tools/fixtures/olala-differential-attention/reference.py /path/to/modeling_olala.py
```

This isolates the attention core with supplied Q/K/V and lambda values; it
does not qualify the projection graph, KV lifecycle or complete Olala model.
