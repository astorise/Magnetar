# Tiny GLM checkpoints

`make_reference.py` creates two synthetic GLM-4.7-shaped Safetensors bundles
and their independent logits with PyTorch 2.15.0 and Transformers 5.23.0.
Both have a dense first layer, a grouped sigmoid-routed second layer, shared
experts and latent attention. They differ in whether one or two experts are
selected; both normalize the selected scores. The query/KV bottleneck RMS
epsilon (1e-6) intentionally differs from the decoder-wide epsilon (1e-5).

The integration tests load these bundles through the normal Hugging Face
loader and the WebAssembly model component. They compare prefill and five
cached decode steps against the recorded logits on the reference and native
CPU providers. These fixtures do not qualify the full published checkpoint
or CUDA routing.
