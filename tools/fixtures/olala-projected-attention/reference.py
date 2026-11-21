"""Generate full projected attention cases from pinned Olala source.

Usage: python reference.py /path/to/modeling_olala.py
"""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import sys

import torch

helpers = runpy.run_path(str(Path(__file__).resolve().parents[1] / "olala-differential-attention/reference.py"))
upstream, Cache, flat = helpers["upstream"], helpers["Cache"], helpers["flat"]


def main(path):
    torch.manual_seed(78624)
    config = SimpleNamespace(num_attention_heads=8, num_signal_heads_diff=6,
                             hidden_size=5, head_dim=4, tpa_rank=3,
                             qk_norm=True, normalization_type="rmsnorm", norm_epsilon=1e-6,
                             zero_centered_gamma=True, softcap_attn=1.4, scalable_softmax=True,
                             cosnet=False, token_shift_attn=True, scalar_proj_as_hidden_matrix=False,
                             rope_theta=0.0, rope_type="", slw_wsize=3, use_completed_p=True,
                             intra_doc_masking=False, xsa=False)
    model = upstream(path)["OlalaDifferentialTensorProductAttentionV2"](config, layer_idx=0).float().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn_like(parameter) * (0.15 if "norm" in name else 0.5))
        hidden = torch.randn(1, 7, config.hidden_size)
        positions = torch.arange(7).view(1, 7)

        def run(start, end, cache):
            output = model(hidden[:, start:end], position_ids=positions[:, start:end], cache_params=cache)[0]
            return {"start": start, "end": end, "output": flat(output),
                    "raw_key": flat(cache.raw_key), "raw_value": flat(cache.raw_value),
                    "keys": flat(cache.key), "values": flat(cache.value)}

        full = run(0, 7, Cache())
        cache = Cache()
        chunks = [run(start, end, cache) for start, end in [(0, 2), (2, 4), (4, 5), (5, 7)]]
        parameters = dict(model.named_parameters())
        matrix = lambda name: flat(parameters[name + ".weight"].t())
        fixture = {
            "hidden_size": config.hidden_size, "query_heads": model.num_attention_heads,
            "kv_heads": model.num_key_value_heads, "signal_heads": model.num_signal_heads,
            "head_dim": config.head_dim, "rank": config.tpa_rank,
            "epsilon": config.norm_epsilon, "window_left": config.slw_wsize,
            "softcap": config.softcap_attn, "scale": 1.0 / config.head_dim,
            "hidden": flat(hidden),
            "weights": {"query": matrix("c_q"), "a_key": matrix("W_A_k"),
                        "b_key": matrix("W_B_k"), "a_value": matrix("W_A_v"),
                        "b_value": matrix("W_B_v"), "shift_key": matrix("shift_proj_k"),
                        "shift_value": matrix("shift_proj_v"), "lambda": matrix("lambda_proj"),
                        "q_norm": flat(parameters["q_norm.norm.weight"]),
                        "k_norm": flat(parameters["k_norm.norm.weight"]),
                        "softmax_scaler": flat(parameters["softmax_scaler"])},
            "full": full, "chunks": chunks,
        }
    destination = Path(__file__).resolve().parents[3] / "magnetar-runtime/fixtures/olala-projected-attention-reference.json"
    destination.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main(sys.argv[1])
