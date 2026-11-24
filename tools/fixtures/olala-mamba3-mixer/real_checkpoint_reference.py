"""Run the pinned Olala Mamba-3 layer using real weights and f32 arithmetic.

Usage: python real_checkpoint_reference.py SOURCE HEADER F32_RANGE BF16_RANGE OUT_DIR

The source and two ranges must match their recorded hashes. The ranges are
relative to the Safetensors data section: [24, 344472) and
[1000145952, 1022707232). The generated binary fixtures and model weights
must remain outside Git. This does not qualify BF16 execution or the full model.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

from reference import upstream

HASHES = {
    "header": "81df2fdc92cc4faa6c66b9786d2103045c6b466a575b31f97151d64f653882f5",
    "f32": "63bc9130e58e93d8f2682b8ca8d676b474ec070dd91a1a5b9bce9c0f8edd7bc7",
    "bf16": "37becb7237e9f3d731b106eeda23aab33a83e3ee315cb8c616da94b0bf642f0f",
}


def read_checked(path, kind):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != HASHES[kind]:
        raise ValueError(f"unexpected {kind} digest")
    return data


def save(directory, name, value):
    data = value.detach().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
    (directory / f"{name}.f32").write_bytes(data.tobytes())


def save_step(directory, name, output, cache):
    save(directory, f"{name}.output", output)
    for suffix, attribute in [
        ("angle", "mamba3_angle_states"),
        ("ssm", "mamba3_ssm_states"),
        ("key", "mamba3_k_states"),
        ("input", "mamba3_v_states"),
    ]:
        save(directory, f"{name}.{suffix}", getattr(cache, attribute)[0])


def main(source, header_path, f32_path, bf16_path, output_path):
    if sys.byteorder != "little":
        raise RuntimeError("reference generation requires a little-endian host")
    header = json.loads(read_checked(header_path, "header")[8:])
    regions = [(24, read_checked(f32_path, "f32")),
               (1000145952, read_checked(bf16_path, "bf16"))]
    config = SimpleNamespace(hidden_size=1536, mamba_mimo_dim=4, mamba_d_state=128,
                             mamba_headdim=64, mamba_ngroups=1,
                             normalization_type="rmsnorm", norm_epsilon=1e-6,
                             zero_centered_gamma=True, cosnet=False,
                             mamba3_postgate_norm=False, complete_slw=False, slw_wsize=32768)
    torch.set_num_threads(1)
    model = upstream(source)(config, layer_idx=0).float().eval()
    prefix = "model.layers.0.mixer."
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
    inputs = (((torch.arange(4 * config.hidden_size, dtype=torch.int64) * 17) % 101 - 50)
              .float().reshape(1, 4, config.hidden_size) / 200.0)
    directory = Path(output_path)
    directory.mkdir(parents=True, exist_ok=True)
    cache = SimpleNamespace(mamba3_angle_states=[None], mamba3_ssm_states=[None],
                            mamba3_k_states=[None], mamba3_v_states=[None], past_length=[0])
    with torch.inference_mode():
        full = model(inputs)[0]
        prefill = model(inputs[:, :2], cache_params=cache)[0]
        save(directory, "input", inputs)
        save(directory, "full", full)
        save_step(directory, "prefill", prefill, cache)
        for index in range(2, 4):
            output = model(inputs[:, index:index + 1], cache_params=cache)[0]
            save_step(directory, f"decode{index - 2}", output, cache)
    print("Wrote real-weight f32 reference for full prefill, chunk and two decode steps")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
