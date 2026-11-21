# Olala token-shift reference

`reference.py` extracts the actual tensor-product projections and token-shift
block from the pinned upstream attention class, stopping before Q/K
normalization. It uses deterministic synthetic weights and records full,
chunked and single-token results, raw states and document boundaries.

Source: [`modeling_olala.py` at revision `b40f8165ee2254c18601220e19842600c6d91794`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/modeling_olala.py),
SHA-256 `357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf`.
The checked-in fixture was generated with PyTorch 2.15.0:

```text
python tools/fixtures/olala-token-shift/reference.py /path/to/modeling_olala.py
```

This isolates token shifts; it does not qualify attention, softmax, score
softcap, or Runtime-managed state for the complete Olala model.
