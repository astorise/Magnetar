# Tiny OLMo 3-shaped fixture

`tiny/` contains generated random projection weights and locally generated
normalization weights, not a released checkpoint. Its full-attention f32 logits
come from `make_reference.py`, an independent NumPy implementation of the
published OLMo 3 decoder equations. It recomputes the full sequence for each of
six greedy steps. Regenerate with `python make_reference.py` and NumPy; no
weights are downloaded.
