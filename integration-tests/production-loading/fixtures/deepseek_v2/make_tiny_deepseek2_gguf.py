"""Turns a tiny DeepSeek-V2 bundle (`make_tiny_deepseek_v2.py`, a `wide`
variant) into a quantized `deepseek2` GGUF file laid out like llama.cpp's own
(the metadata keys and tensor names of `MODEL_ARCH.DEEPSEEK2`), plus a
reference bundle whose weights are the *dequantized* values, so
`transformers` computes on exactly the numbers the quantized file holds.

Two forms of the latent attention's up-projection:
  kv_b    `attn_kv_b`, as llama.cpp's converter wrote it until 2025: one
          [heads * (nope + v), latent] matrix; `head_count_kv` is the head
          count and `key_length`/`value_length` the head widths. No
          `expert_gating_func` and no YaRN betas, as those files.
  split   `attn_k_b` [heads, latent, nope] (each head's key rows transposed)
          and `attn_v_b` [heads, v, latent], as the converter writes it
          since: `head_count_kv` 1, `key_length`/`value_length` the
          latent's widths, `key_length_mla`/`value_length_mla` the heads'.

Usage: make_tiny_deepseek2_gguf.py <bundle_dir> <out_dir> --form kv_b|split
           [--proj q8_0|q4_0] [--experts q8_0|q4_0] [--embd q8_0|q4_0|f32]
  <out_dir>/gguf/model.gguf   the quantized file
  <out_dir>/ref/              config.json + model.safetensors (dequantized)
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from gguf import GGMLQuantizationType as T
from gguf import ExpertGatingFuncType, GGUFWriter
from gguf.quants import dequantize, quantize
from safetensors.torch import load_file, save_file

QUANT = {"q4_0": T.Q4_0, "q8_0": T.Q8_0, "f32": T.F32}
ATTENTION = {"q_proj": "attn_q", "q_a_proj": "attn_q_a", "q_b_proj": "attn_q_b",
             "kv_a_proj_with_mqa": "attn_kv_a_mqa", "o_proj": "attn_output"}
NORMS = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
         "self_attn.q_a_layernorm": "attn_q_a_norm", "self_attn.kv_a_layernorm": "attn_kv_a_norm"}
MLP = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("out")
    ap.add_argument("--form", choices=["kv_b", "split"], required=True)
    ap.add_argument("--proj", default="q8_0")
    ap.add_argument("--experts", default="q4_0")
    ap.add_argument("--embd", default="q8_0")
    args = ap.parse_args()

    c = json.load(open(f"{args.bundle}/config.json"))
    tensors = load_file(f"{args.bundle}/model.safetensors")
    n, heads = c["num_hidden_layers"], c["num_attention_heads"]
    nope, rope, v, latent = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"], c["kv_lora_rank"]
    experts = c["n_routed_experts"]
    split = args.form == "split"

    os.makedirs(f"{args.out}/gguf", exist_ok=True)
    os.makedirs(f"{args.out}/ref", exist_ok=True)
    w = GGUFWriter(f"{args.out}/gguf/model.gguf", "deepseek2")
    w.add_name("tiny-deepseek2")
    w.add_context_length(c["max_position_embeddings"])
    w.add_embedding_length(c["hidden_size"])
    w.add_feed_forward_length(c["intermediate_size"])
    w.add_block_count(n)
    w.add_head_count(heads)
    w.add_head_count_kv(1 if split else heads)
    rp = c["rope_parameters"]
    w.add_rope_freq_base(rp["rope_theta"])
    w.add_layer_norm_rms_eps(c["rms_norm_eps"])
    w.add_expert_used_count(c["num_experts_per_tok"])
    if split:
        w.add_expert_gating_func(ExpertGatingFuncType.SOFTMAX)
    w.add_leading_dense_block_count(c["first_k_dense_replace"])
    w.add_vocab_size(c["vocab_size"])
    if c.get("q_lora_rank"):
        w.add_q_lora_rank(c["q_lora_rank"])
    w.add_kv_lora_rank(latent)
    if split:
        w.add_key_length(latent + rope)
        w.add_value_length(latent)
        w.add_key_length_mla(nope + rope)
        w.add_value_length_mla(v)
    else:
        w.add_key_length(nope + rope)
        w.add_value_length(v)
    w.add_expert_feed_forward_length(c["moe_intermediate_size"])
    w.add_expert_count(experts)
    w.add_expert_shared_count(c["n_shared_experts"])
    w.add_expert_weights_scale(c["routed_scaling_factor"])
    w.add_rope_dimension_count(rope)
    if rp.get("rope_type") == "yarn":
        if rp["mscale"] != rp["mscale_all_dim"]:
            raise SystemExit("a GGUF file holds one YaRN multiplier: mscale must equal mscale_all_dim")
        w.add_string("deepseek2.rope.scaling.type", "yarn")
        w.add_rope_scaling_factor(rp["factor"])
        w.add_rope_scaling_orig_ctx_len(rp["original_max_position_embeddings"])
        if split:
            w.add_rope_scaling_yarn_beta_fast(rp["beta_fast"])
            w.add_rope_scaling_yarn_beta_slow(rp["beta_slow"])
        elif (rp["beta_fast"], rp["beta_slow"]) != (32.0, 1.0):
            raise SystemExit("the kv_b form writes no betas: the bundle must use 32 and 1")
        w.add_rope_scaling_yarn_log_mul(0.1 * rp["mscale_all_dim"])
    vocab = c["vocab_size"]
    # No beginning-of-sequence token: the Runtime never samples one after
    # the first step, while the greedy references may pick any token.
    w.add_uint32("tokenizer.ggml.eos_token_id", c["eos_token_id"])
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "ab"] + [f"t{i}" for i in range(3, vocab)])
    # One merge that yields a vocabulary token; the tests pass token ids.
    w.add_array("tokenizer.ggml.merges", ["a b"])

    ref = {}

    def store(gguf_name, x, kind):
        """Writes `x` and returns the values the file holds."""
        x = x.astype(np.float32)
        if kind == "f32" or x.ndim < 2:
            w.add_tensor(gguf_name, x)
            return x
        q = quantize(x, QUANT[kind])
        w.add_tensor(gguf_name, q, raw_dtype=QUANT[kind])
        return dequantize(q, QUANT[kind]).astype(np.float32).reshape(x.shape)

    def put(hf_name, gguf_name, kind):
        ref[hf_name] = torch.from_numpy(store(gguf_name, tensors[hf_name].float().numpy(), kind))

    put("model.embed_tokens.weight", "token_embd.weight", args.embd)
    put("model.norm.weight", "output_norm.weight", "f32")
    put("lm_head.weight", "output.weight", args.proj)
    for i in range(n):
        p, b = f"model.layers.{i}.", f"blk.{i}."
        for hf, g in NORMS.items():
            if p + hf + ".weight" in tensors:
                put(p + hf + ".weight", b + g + ".weight", "f32")
        for hf, g in ATTENTION.items():
            if p + "self_attn." + hf + ".weight" in tensors:
                put(p + "self_attn." + hf + ".weight", b + g + ".weight", args.proj)
        kv_b = p + "self_attn.kv_b_proj.weight"
        if split:
            per_head = tensors[kv_b].float().numpy().reshape(heads, nope + v, latent)
            k_b = store(b + "attn_k_b.weight", per_head[:, :nope].transpose(0, 2, 1).copy(), args.proj)
            v_b = store(b + "attn_v_b.weight", per_head[:, nope:].copy(), args.proj)
            joined = np.concatenate([k_b.transpose(0, 2, 1), v_b], axis=1)
            ref[kv_b] = torch.from_numpy(joined.reshape(heads * (nope + v), latent).copy())
        else:
            put(kv_b, b + "attn_kv_b.weight", args.proj)
        if i < c["first_k_dense_replace"]:
            for hf, g in MLP.items():
                put(p + f"mlp.{hf}.weight", b + f"ffn_{g}.weight", args.proj)
            continue
        put(p + "mlp.gate.weight", b + "ffn_gate_inp.weight", "f32")
        for hf, g in MLP.items():
            put(p + f"mlp.shared_experts.{hf}.weight", b + f"ffn_{g}_shexp.weight", args.proj)
            names = [p + f"mlp.experts.{e}.{hf}.weight" for e in range(experts)]
            stacked = np.stack([tensors[name].float().numpy() for name in names])
            held = store(b + f"ffn_{g}_exps.weight", stacked, args.experts)
            for e, name in enumerate(names):
                ref[name] = torch.from_numpy(held[e].copy())
    unmapped = set(tensors) - set(ref)
    if unmapped:
        raise SystemExit(f"unmapped tensors {sorted(unmapped)}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    save_file({k: t.contiguous() for k, t in ref.items()}, f"{args.out}/ref/model.safetensors",
              metadata={"format": "pt"})
    for name in ("config.json", "generation_config.json"):
        if os.path.exists(f"{args.bundle}/{name}"):
            shutil.copy(f"{args.bundle}/{name}", f"{args.out}/ref/{name}")
    print("wrote", f"{args.out}/gguf/model.gguf", os.path.getsize(f"{args.out}/gguf/model.gguf"), "bytes")


if __name__ == "__main__":
    main()
