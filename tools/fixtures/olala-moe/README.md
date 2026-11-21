`reference.py` extracts the verified `OlalaMoE`, `TorchMoEMLP` and `OlalaMLP`
definitions from the [pinned upstream model](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/modeling_olala.py).
It generates synthetic top-1 and top-2 examples with routed bottleneck
projections, ungated ReLU² experts, shared ReLU² MLP and a linear logits head.
No upstream model weights are distributed. Run with PyTorch 2.15:

```sh
python tools/fixtures/olala-moe/reference.py /path/to/modeling_olala.py
```

The output is `magnetar-runtime/fixtures/olala-moe-reference.json`, so the
Runtime's tests and published crate can both access it.
