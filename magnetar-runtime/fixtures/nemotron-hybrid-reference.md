# Synthetic mixed Nemotron-H reference

`nemotron-hybrid-reference.json` contains deterministic synthetic weights,
token IDs, logits and states. The generator executes the actual PyTorch
Mamba-2 mixer, attention, dense MLP, router, MoE and RMSNorm definitions from
the pinned NVIDIA model source (SHA-256
`52273a23a2c768ffe5938e96f6832afd762029293a8be9bfed886757ba6a39b5`), and
the gated normalization reference from mamba-ssm v2.2.4 (SHA-256
`eb6252e247b90f1c8a75946efbc1a221e0c4da701b6757ddae49f3495cf7a42f`).
It uses a cache shim for the source's inconsistent cache interface. Its
composition passes attention history explicitly; the pinned whole-model block
does not. This fixture therefore validates that explicit composition, not
cached equivalence to the unmodified whole-model block.

The real Wasmtime Component builds a four-block graph for these weights. The
Reference CPU test compares full prefill, a shorter prefill followed by four
cached steps, convolution and SSM states, attention K/V, and a new session
after release. It also rejects unsupported position settings and missing
weights. The fixture does not contain checkpoint weights and does not qualify
checkpoint loading, native BF16 arithmetic or GPU execution.

Reproduce with PyTorch, Transformers and einops, supplying the two verified
source files:

```sh
python integration-tests/production-loading/fixtures/generate_nemotron_hybrid_reference.py \
  /path/to/modeling_nemotron_h.py /path/to/layernorm_gated.py \
  magnetar-runtime/fixtures/nemotron-hybrid-reference.json
```
