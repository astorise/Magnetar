# Tiny Kimi K2 FP8 checkpoints

`make_reference.py` generates two synthetic Kimi K2-shaped checkpoints and
logits using PyTorch 2.15.0, Transformers 4.48.3, and the original Kimi model
code from revision `22bf139598edbbbcd9226c6ea392200d4b53f4b6`. It checks the
SHA-256 of the upstream model and configuration modules before importing them.
The examples cover top-1 and top-2 sigmoid routing, YaRN latent attention,
one FP8 latent projection and one FP8 expert projection with 128×128 block
scales. The model used for the oracle receives the independently decoded
weights, and the saved bundle includes the nonpersistent RoPE buffers found
in the public checkpoint.

The production-loading tests compare prefill and cached decode logits on the
CPU providers. These small synthetic fixtures do not qualify the complete
1.03 TB public checkpoint or native FP8 GPU execution.
