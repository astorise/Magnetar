"""Run llama.cpp's CLI with a local GGUF model."""
import re
from pathlib import Path
from common import PROMPT, sha256, timed_command


def run(args, raw_dir):
    if not args.binary or not args.model_path or not Path(args.model_path).is_file():
        raise ValueError("llama.cpp requires --binary and --model-path (GGUF file)")
    model = Path(args.model_path).resolve()
    command = [str(Path(args.binary).resolve()), "--model", str(model), "--prompt", PROMPT,
               "--n-predict", str(args.tokens), "--temp", "0", "--repeat-penalty", "1",
               "--n-gpu-layers", "0" if args.device == "cpu" else "999",
               "-no-cnv", "--perf"]
    raw_dir.mkdir(parents=True, exist_ok=False)
    samples = []
    for index in range(args.runs + 1):
        stdout, stderr, elapsed = timed_command(command)
        log = stdout + "\n--- stderr ---\n" + stderr
        (raw_dir / f"run-{index}.txt").write_text(log, encoding="utf-8")
        matches = re.findall(r"(?m)^.*?(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+(?:runs|tokens).*?([\d.]+)\s+tokens per second", log)
        if not matches:
            raise RuntimeError("llama.cpp did not report eval timings; inspect raw log")
        offload = re.search(r"offloaded\s+(\d+)\s*/\s*\d+\s+layers", log, re.IGNORECASE)
        if args.device == "gpu" and offload and int(offload[1]) == 0:
            raise RuntimeError("llama.cpp reported zero GPU layers; inspect raw log")
        ms, count, rate = matches[-1]
        if index:
            samples.append({"decode_tokens": int(count), "decode_ms": float(ms),
                            "decode_tokens_per_second": float(rate), "process_seconds_including_load": elapsed})
    return {"protocol": "one warmup process, then fresh process and model load per measured run",
            "device_evidence": "requested GPU offload; inspect raw logs for actual offloaded layers" if args.device == "gpu" else "GPU layers set to 0",
            "model": {"path": str(model), "weights_sha256": sha256(model)},
            "dtype": "GGUF, inspect model metadata", "command": command, "samples": samples}
