# Tiny Mistral-shaped fixtures

These fixtures reuse the generated random weights of `../smollm2/tiny`; they
are not released Mistral checkpoint weights. `full` uses the existing
Transformers reference for the equivalent full-attention decoder. `window3`
uses an independent NumPy implementation of causal three-token sliding
attention, full-sequence recomputation and six greedy steps. The eight-token
prompt and cached decode cross the window boundary.

`make_reference.py` first verifies its NumPy full-attention logits against the
checked-in Transformers reference before producing either fixture. Regenerate
with `python make_reference.py` and NumPy installed. No weights are downloaded.
