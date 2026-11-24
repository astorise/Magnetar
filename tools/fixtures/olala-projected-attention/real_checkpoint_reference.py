"""Run pinned Olala attention with real weights and synthetic f32 inputs.

Usage: python real_checkpoint_reference.py SOURCE HEADER F32_RANGE BF16_RANGE OUT_DIR

Data-section ranges are [55829544, 55829736) and
[11481542688, 11503968800). Weights and generated files must remain outside
Git. This tests the attention class, not full-model or native BF16 inference.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from reference import upstream, Cache

HASHES = {
    "header": "81df2fdc92cc4faa6c66b9786d2103045c6b466a575b31f97151d64f653882f5",
    "f32": "589d98523668c2226beea57b4e69d073b8bd88d686672a32b1798d07c03366eb",
    "bf16": "90c8e3ac8d85c7e17d0bb8370d9d41cd66fa656996ab0322a42b467a8e94d03f",
}


def checked(path, kind):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != HASHES[kind]:
        raise ValueError(f"unexpected {kind} digest")
    return data


def save(directory, name, value):
    data = value.detach().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
    (directory / f"{name}.f32").write_bytes(data.tobytes())


def run(directory, name, model, hidden, positions, cache):
    output = model(hidden, position_ids=positions, cache_params=cache)[0]
    for suffix, value in [("output", output), ("raw_key", cache.raw_key),
                          ("raw_value", cache.raw_value), ("keys", cache.key),
                          ("values", cache.value)]:
        save(directory, f"{name}.{suffix}", value)


def main(source, header_path, f32_path, bf16_path, output_path):
    if sys.byteorder != "little":
        raise RuntimeError("reference generation requires a little-endian host")
    header = json.loads(checked(header_path, "header")[8:])
    regions = [(55829544, checked(f32_path, "f32")),
               (11481542688, checked(bf16_path, "bf16"))]
    config = SimpleNamespace(num_attention_heads=48, num_signal_heads_diff=36,
                             hidden_size=1536, head_dim=128, tpa_rank=4,
                             qk_norm=True, normalization_type="rmsnorm", norm_epsilon=1e-6,
                             zero_centered_gamma=True, softcap_attn=150.0, scalable_softmax=True,
                             cosnet=False, token_shift_attn=True, scalar_proj_as_hidden_matrix=True,
                             rope_theta=0.0, rope_type="", slw_wsize=32768, use_completed_p=True,
                             intra_doc_masking=False, xsa=False)
    torch.set_num_threads(1)
    # Layer index affects only the cache key inside this class. Using 0 allows
    # reuse of the small test cache while loading layer 4's actual weights.
    model = upstream(source)["OlalaDifferentialTensorProductAttentionV2"](
        config, layer_idx=0).float().eval()
    prefix = "model.layers.4.mixer."
    state = {}
    for name in model.state_dict():
        entry = header[prefix + name]
        start, end = entry["data_offsets"]
        data = next((data[start - base:end - base] for base, data in regions
                     if base <= start and end <= base + len(data)), None)
        if data is None:
            raise ValueError(f"missing bytes for {name}")
        dtype = {"F32": torch.float32, "BF16": torch.bfloat16}[entry["dtype"]]
        state[name] = torch.frombuffer(bytearray(data), dtype=dtype).reshape(entry["shape"]).float()
    model.load_state_dict(state, strict=True)
    hidden = (((torch.arange(4 * config.hidden_size, dtype=torch.int64) * 17) % 101 - 50)
              .float().reshape(1, 4, config.hidden_size) / 200.0)
    positions = torch.arange(4).view(1, 4)
    directory = Path(output_path)
    directory.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        save(directory, "input", hidden)
        run(directory, "full", model, hidden, positions, Cache())
        cache = Cache()
        run(directory, "prefill", model, hidden[:, :2], positions[:, :2], cache)
        for index in range(2, 4):
            run(directory, f"decode{index - 2}", model,
                hidden[:, index:index + 1], positions[:, index:index + 1], cache)
    print("Wrote real-weight attention reference for full prefill, prefix and two decode steps")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
