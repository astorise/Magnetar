# Synthetic block-FP8 reference fixtures

`tiny` is based on the synthetic SmolLM2-shaped Llama fixture; one Q
projection is stored as E4M3FN with F32 scales. `deepseek` is based on the
synthetic DeepSeek V2 fixture; one latent-attention matrix and one routed
expert matrix are stored in the same format. Both use 7-by-7 blocks so the
last blocks are partial. The source fixtures contain random test weights.

`make_reference.py` and `make_deepseek_reference.py` generate the bundles and
independent Transformers logits with PyTorch 2.15.0 and Transformers 5.23.0.
The reference model multiplies decoded E4M3FN values by their scale in F32;
it does not simulate dynamic activation quantization or native FP8 kernels.
These fixtures do not qualify the released Kimi checkpoint or GPU execution.
