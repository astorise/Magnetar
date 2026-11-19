"""Run Candle's Qwen example, once per process."""
import re
from pathlib import Path
from common import PROMPT, sha256, timed_command


def run(args, raw_dir):
    if not args.binary or not args.model_path:
        raise ValueError("Candle requires --binary and --model-path (local Qwen bundle)")
    model = Path(args.model_path).resolve()
    weights = model / "model.safetensors"
    if not weights.is_file():
        raise ValueError("Candle adapter requires a single model.safetensors")
    command = [str(Path(args.binary).resolve()), "--model", "2-0.5b", "--weight-path", str(model),
               "--tokenizer-file", str(model / "tokenizer.json"), "--prompt", PROMPT,
               "--sample-len", str(args.tokens), "--repeat-penalty", "1", "--no-chat-template"]
    if args.device == "cpu":
        command.append("--cpu")
    raw_dir.mkdir(parents=True, exist_ok=False)
    samples = []
    for index in range(args.runs + 1):
        stdout, stderr, elapsed = timed_command(command, cwd=model)
        (raw_dir / f"run-{index}.txt").write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
        if args.device == "gpu" and "Running on CPU" in stdout:
            raise RuntimeError("Candle fell back to CPU; build with CUDA or Metal")
        match = re.search(r"(\d+) tokens generated \(([\d.]+) token/s\)", stdout)
        if not match:
            raise RuntimeError("Candle output did not contain generation metrics")
        count, rate = int(match[1]), float(match[2])
        if count < 1 or rate <= 0:
            raise RuntimeError("Candle returned a non-positive token count or generation rate")
        if index:
            # Candle prints only a rounded rate. Its timer starts after model load
            # and includes prompt prefill and generation, so this duration is an estimate.
            samples.append({"generated_tokens": count, "generation_tokens_per_second": rate,
                            "request_seconds": count / rate, "request_tokens_per_second": rate,
                            "process_seconds_including_load": elapsed})
    return {"protocol": "one warmup process, then fresh process and model load per measured run; "
                        "request duration estimated from Candle's rounded prefill-plus-generation rate",
            "device_evidence": "requested; inspect raw output and build features",
            "model": {"weights_sha256": sha256(weights), "path": str(model)},
            "dtype": "F32 CPU; BF16 GPU in the Candle example", "command": command, "samples": samples}
