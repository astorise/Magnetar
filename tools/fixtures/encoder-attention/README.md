`reference.py` extracts the actual ModernBERT eager attention function and
bidirectional padding/window mask functions from Transformers 5.23.0. The
sources are verified by SHA-256 before use:

- `modeling_modernbert.py`: `29b598c408582bd53c0a48ae34dce2cb008aefc5f6087b03c310b5ce7bb0de9b`
- `masking_utils.py`: `fc05cdaf4cc8c93ab5cded94cea88ee79b465984915a43057093ea6b958b6268`

It generates synthetic global,
padded, local-window, self-only and custom-scale cases; no model weights are
included. Positions and Q/K rotary encoding are inputs to this operator, not
computed by the fixture.

```sh
python tools/fixtures/encoder-attention/reference.py MODELING_MODERNBERT.py MASKING_UTILS.py
```

The output is `magnetar-runtime/fixtures/encoder-attention-reference.json`.
