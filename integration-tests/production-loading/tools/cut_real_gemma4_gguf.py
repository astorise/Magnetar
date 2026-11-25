"""Cuts the real `gemma-4-12B-it` Q4_0 GGUF to a few of its layers and its first
vocabulary rows, keeping every kept tensor's bytes as they are, and writes the
float32 *dequantized* reference bundle `transformers` computes on
(`components/gemma/tools/qualification/gemma4_reference_logits.py`).

The cut file is small enough to compare against `transformers` on a 15 GB host,
which the 12B itself is not: the 7.2 GB file dequantizes to 48 GB. Comparing
Magnetar's run of the cut with the reference tests the Gemma 4 graph on the
real weights, norms, layer scales and `rope_freqs`, not the quantization error.

Usage (the Python environment of the qualification scripts: torch, safetensors,
the `gguf` package; run it with `python -I` on a downloaded file):

  python -I cut_real_gemma4_gguf.py <real model.gguf> <source layers> <vocab rows> <out dir>

  <source layers>  comma-separated indices of the real layers to keep, in order
                   (a layer is a full-attention layer when index % 6 == 5)
  <out dir>/gguf/model.gguf   the cut file
  <out dir>/ref/              config.json + model.safetensors, for the reference

`G_OUT_SCALE` (default 1) scales `output_norm.weight` in both: a cut of three
layers saturates the logit soft-cap at 30, which hides differences, so the
comparison scales the final norm down (0.01) to keep the logits near 2.
"""
import json, os, sys
import numpy as np
import torch
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize
from safetensors.torch import save_file

real, sel, rows, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]), sys.argv[4]
layers = len(sel)
os.makedirs(f"{out}/gguf", exist_ok=True)
os.makedirs(f"{out}/ref", exist_ok=True)
reader = GGUFReader(real)
tensors = {t.name: t for t in reader.tensors}
kinds = [((i + 1) % 6 != 0) for i in sel]  # the 12B's pattern: five sliding layers, then a full one
PROJ = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj"}
NORM = {"attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
        "ffn_norm": "pre_feedforward_layernorm", "post_ffw_norm": "post_feedforward_layernorm",
        "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}
PREFIX = "model.language_model."

w = GGUFWriter(f"{out}/gguf/model.gguf", "gemma4")
w.add_string("general.name", "gemma-4-12B-it-cut")
a = "gemma4."
w.add_uint32(a + "context_length", 262144)
w.add_uint32(a + "embedding_length", 3840)
w.add_uint32(a + "feed_forward_length", 15360)
w.add_uint32(a + "block_count", layers)
w.add_uint32(a + "attention.head_count", 16)
w.add_array(a + "attention.head_count_kv", [8 if k else 1 for k in kinds])
w.add_float32(a + "rope.freq_base", 1000000.0)
w.add_float32(a + "rope.freq_base_swa", 10000.0)
w.add_float32(a + "attention.layer_norm_rms_epsilon", 9.999999974752427e-07)
w.add_uint32(a + "attention.key_length", 512)
w.add_uint32(a + "attention.value_length", 512)
w.add_float32(a + "final_logit_softcapping", 30.0)
w.add_uint32(a + "attention.sliding_window", 1024)
w.add_uint32(a + "attention.shared_kv_layers", 0)
w.add_uint32(a + "embedding_length_per_layer_input", 0)
w.add_array(a + "attention.sliding_window_pattern", kinds)
w.add_uint32(a + "attention.key_length_swa", 256)
w.add_uint32(a + "attention.value_length_swa", 256)
w.add_uint32(a + "rope.dimension_count", 512)
w.add_uint32(a + "rope.dimension_count_swa", 256)
w.add_uint32("tokenizer.ggml.bos_token_id", 2)
w.add_uint32("tokenizer.ggml.eos_token_id", 1)
w.add_string("tokenizer.ggml.model", "gpt2")
w.add_array("tokenizer.ggml.tokens", ["a", "b", "ab"] + [f"t{i}" for i in range(3, rows)])
w.add_array("tokenizer.ggml.merges", ["a b"])

ref = {}
def add(gguf_name, hf_name, row_limit=None, dst=None):
    t = tensors[gguf_name]
    dst = dst or gguf_name
    data = np.array(t.data)
    if row_limit is not None:
        data = data[:row_limit]
    if t.tensor_type.name == "F32":
        arr = data.astype(np.float32).reshape(data.shape)
        if gguf_name == "output_norm.weight":
            arr = arr * np.float32(float(os.environ.get("G_OUT_SCALE", "1")))
        w.add_tensor(dst, arr)
        ref[hf_name] = torch.from_numpy(arr.copy())
    else:
        w.add_tensor(dst, data, raw_dtype=t.tensor_type)
        ref[hf_name] = torch.from_numpy(dequantize(data, t.tensor_type).astype(np.float32))

add("token_embd.weight", PREFIX + "embed_tokens.weight", rows)
add("output_norm.weight", PREFIX + "norm.weight")
add("rope_freqs.weight", "rope_freqs.weight")
ref.pop("rope_freqs.weight")
for j, i in enumerate(sel):
    for g, h in PROJ.items():
        if f"blk.{i}.{g}.weight" in tensors:
            add(f"blk.{i}.{g}.weight", f"{PREFIX}layers.{j}.{h}.weight", dst=f"blk.{j}.{g}.weight")
    for g, h in NORM.items():
        add(f"blk.{i}.{g}.weight", f"{PREFIX}layers.{j}.{h}.weight", dst=f"blk.{j}.{g}.weight")
    add(f"blk.{i}.layer_output_scale.weight", f"{PREFIX}layers.{j}.layer_scalar", dst=f"blk.{j}.layer_output_scale.weight")
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
for k, v in list(ref.items()):
    if k.endswith("layer_scalar"):
        ref[k] = v.reshape(-1)
save_file({k: v.contiguous() for k, v in ref.items()}, f"{out}/ref/model.safetensors", metadata={"format": "pt"})

cfg = dict(
    attention_bias=False, attention_dropout=0.0, attention_k_eq_v=True, bos_token_id=2, enable_moe_block=False,
    eos_token_id=1, final_logit_softcapping=30.0, global_head_dim=512, head_dim=256,
    hidden_activation="gelu_pytorch_tanh", hidden_size=3840, hidden_size_per_layer_input=0,
    initializer_range=0.02, intermediate_size=15360,
    layer_types=["sliding_attention" if k else "full_attention" for k in kinds],
    max_position_embeddings=4096, model_type="gemma4_text", num_attention_heads=16,
    num_global_key_value_heads=1, num_hidden_layers=layers, num_key_value_heads=8, num_kv_shared_layers=0,
    pad_token_id=0,
    rms_norm_eps=1e-6,
    rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                     "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    sliding_window=1024, tie_word_embeddings=True, use_bidirectional_attention=None, use_cache=True,
    use_double_wide_mlp=False, vocab_size=rows, vocab_size_per_layer_input=rows)
bundle = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "tie_word_embeddings": True,
          "dtype": "float32", "eos_token_id": [1, 106],
          "text_config": {k: v for k, v in cfg.items() if k != "model_type"} | {"model_type": "gemma4_text", "dtype": "float32"}}
json.dump(bundle, open(f"{out}/ref/config.json", "w"), indent=2)
print("wrote", os.path.getsize(f"{out}/gguf/model.gguf"), "bytes GGUF;", len(ref), "reference tensors")
