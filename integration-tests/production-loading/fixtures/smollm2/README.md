# Tiny SmolLM2-shaped checkpoint

`tiny/` is a randomly initialized three-layer Llama checkpoint with tied
embeddings, grouped-query attention, SiLU and the RoPE base used by the
published SmolLM2-135M-Instruct configuration. It is not the released
SmolLM2 weights and cannot establish the quality of that model.

`make_reference.py` creates the weights and obtains six greedy steps from
`transformers.LlamaForCausalLM` by recomputing the whole sequence at each
step. Magnetar's integration test compares those independent f32 logits to a
prefill followed by cached decode, using its real loader, Component and CPU
Provider.

To regenerate with Python, torch 2.15.0, transformers 5.23.0, numpy 2.4.4
and safetensors 0.8.0:

```sh
python make_reference.py tiny
```

The model configuration and weights here are generated, not downloaded from
the published checkpoint. The published architecture reference is
[`HuggingFaceTB/SmolLM2-135M-Instruct` at `12fd25f`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json).
