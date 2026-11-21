# Tiny SmolLM3 reference

This randomly initialized, three-layer checkpoint reuses the generated
SmolLM2 fixture projections and enables RoPE in layers 0 and 2 only. It is not
the released SmolLM3 checkpoint. `make_reference.py` obtains six greedy steps
from `transformers.SmolLM3ForCausalLM`, recomputing the full sequence each
time; Magnetar compares prefill and cached decode with these independent logits.

Regenerate using PyTorch 2.15.0, Transformers 5.23.0, NumPy 2.4.4 and
Safetensors 0.8.0. No model weights are downloaded.
