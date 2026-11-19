"""Benchmark AirLLM in a separate process with a resident model wrapper."""
import json
import sys
from pathlib import Path

from common import ROOT, sha256, timed_command


def run(args, raw_dir):
    if not args.model_path or not Path(args.model_path).is_dir():
        raise ValueError("AirLLM requires a local Hugging Face model directory")
    raw_dir.mkdir(parents=True, exist_ok=False)
    result_path = raw_dir / "airllm.json"
    command = [sys.executable, str(ROOT / "benchmark/airllm/worker.py"),
               "--model-path", str(Path(args.model_path).resolve()), "--device", args.device,
               "--tokens", str(args.tokens), "--runs", str(args.runs), "--output", str(result_path.resolve())]
    stdout, stderr, elapsed = timed_command(command, cwd=ROOT)
    (raw_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (raw_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["model"]["weights_sha256"] = sha256(Path(args.model_path) / "model.safetensors")
    if len(result["samples"]) != args.runs:
        raise RuntimeError("AirLLM did not return the requested number of samples")
    return {**result, "protocol": "resident AirLLM wrapper; one warmup then synchronous generation; weights streamed by AirLLM",
            "command_seconds": elapsed}
