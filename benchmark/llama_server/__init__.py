"""Run llama-server locally for a resident-model comparison."""
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError

from common import PROMPT, http_json, sha256


def run(args, raw_dir):
    if not args.binary or not args.model_path or not Path(args.model_path).is_file():
        raise ValueError("llama-server requires a binary and a GGUF model")
    model = Path(args.model_path).resolve()
    raw_dir.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [str(Path(args.binary).resolve()), "--model", str(model), "--host", "127.0.0.1",
               "--port", str(port), "--parallel", "1", "--ctx-size", "512",
               "--n-gpu-layers", "0" if args.device == "cpu" else "999"]
    experimental = os.environ.get("BENCH_MODEL_PROFILE") == "qwen3.6-35b-experimental"
    if experimental and args.device == "gpu":
        # The 35B MoE model does not fit the runner's 12 GB GPU; keep experts on CPU.
        command.append("--cpu-moe")
    # Recent llama-server versions hide loader/offload messages at the default verbosity.
    if args.device == "gpu":
        command.append("--verbose")
    log_path = raw_dir / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + (900 if experimental else 300)
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited ({process.returncode}); see {log_path}")
                try:
                    if http_json(url + "/health", timeout=2).get("status") == "ok":
                        break
                except (OSError, URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"llama-server did not become ready; see {log_path}")
                time.sleep(1)
            log.flush()
            evidence = log_path.read_text(encoding="utf-8", errors="replace")
            offload = re.search(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers", evidence, re.IGNORECASE)
            if args.device == "gpu" and (not offload or int(offload[1]) == 0):
                raise RuntimeError(f"llama-server did not confirm GPU layer offload; see {log_path}")
            prompt = PROMPT
            if experimental:
                tokenized = http_json(url + "/tokenize", {"content": PROMPT, "add_special": False})
                prompt = tokenized.get("tokens")
                if not isinstance(prompt, list) or not prompt or any(type(t) is not int for t in prompt):
                    raise RuntimeError("llama-server did not return valid prompt token IDs")
                ids_path = os.environ.get("BENCH_PROMPT_IDS_PATH")
                if not ids_path:
                    raise RuntimeError("BENCH_PROMPT_IDS_PATH is required for Qwen3.6")
                Path(ids_path).write_text(json.dumps(prompt), encoding="utf-8")
            body = {"prompt": prompt, "n_predict": args.tokens, "temperature": 0,
                    "repeat_penalty": 1, "cache_prompt": False, "return_tokens": True}
            samples = []
            for index in range(args.runs + 1):
                started = time.perf_counter()
                response = http_json(url + "/completion", body)
                elapsed = time.perf_counter() - started
                (raw_dir / f"run-{index}.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
                timings = response.get("timings") or {}
                generated = timings.get("predicted_n")
                prompt = timings.get("prompt_n")
                if not isinstance(generated, int) or generated < 1 or not isinstance(prompt, int):
                    raise RuntimeError("llama-server response lacks token counts")
                if index:
                    samples.append({"generated_tokens": generated, "prompt_tokens": prompt,
                                    "request_seconds": elapsed, "request_tokens_per_second": generated / elapsed,
                                    "decode_tokens_per_second": timings.get("predicted_per_second"),
                                    "text": response.get("content"), "generated_token_ids": response.get("tokens")})
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return {"protocol": "local resident llama-server; one warmup then non-streaming HTTP requests; prefix cache disabled"
                        + ("; MoE experts kept on CPU" if experimental and args.device == "gpu" else ""),
            "device_evidence": offload.group(0) if offload else "GPU offload disabled",
            "model": {"path": str(model), "weights_sha256": sha256(model)},
            "dtype": "GGUF UD-Q4_K_XL mixed weights" if experimental else "GGUF FP16 weights",
            "command": command, "samples": samples}
