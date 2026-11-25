"""Benchmark a running Ollama server using its native generation API."""
import json
import re
import time
from pathlib import Path
from common import PROMPT, http_json


def discovered_compute(server_log):
    """The compute libraries the Ollama server reported discovering at
    startup (`msg="inference compute" ... library=...`), or None when the
    log is not given, unreadable or reports none."""
    if not server_log:
        return None
    try:
        text = Path(server_log).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    libraries = set(re.findall(r'msg="inference compute".*?\blibrary=([^\s"]+)', text))
    return libraries or None


def canonical_name(name):
    if not isinstance(name, str):
        return None
    name = name.removeprefix("registry.ollama.ai/").removeprefix("library/")
    if ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    return name


def run(args, raw_dir):
    if not args.model:
        raise ValueError("Ollama requires --model naming an installed model")
    base = (args.url or "http://127.0.0.1:11434").rstrip("/")
    body = {"model": args.model, "prompt": PROMPT, "raw": True, "stream": False,
            "keep_alive": "30m", "options": {"temperature": 0, "repeat_penalty": 1,
            "num_predict": args.tokens, "num_gpu": 0 if args.device == "cpu" else 999}}
    raw_dir.mkdir(parents=True, exist_ok=False)
    samples = []
    placement = None
    compute = discovered_compute(getattr(args, "server_log", None))
    for index in range(args.runs + 1):
        started = time.perf_counter()
        result = http_json(base + "/api/generate", body)
        elapsed = time.perf_counter() - started
        (raw_dir / f"run-{index}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not result.get("done") or not isinstance(result.get("eval_count"), int) or result["eval_count"] < 1:
            raise RuntimeError("Ollama returned an incomplete response or no token count")
        ps = http_json(base + "/api/ps")
        (raw_dir / f"placement-{index}.json").write_text(json.dumps(ps, indent=2), encoding="utf-8")
        models = ps.get("models", [])
        names = {canonical_name(args.model), canonical_name(result.get("model"))} - {None}
        placement = next((m for m in models if canonical_name(m.get("name")) in names
                          or canonical_name(m.get("model")) in names), None)
        if placement is None or not isinstance(placement.get("size_vram"), int):
            raise RuntimeError("Cannot verify Ollama model placement via /api/ps")
        vram = placement["size_vram"]
        # On a CPU with AMX, Ollama's CPU backend keeps the weights in a
        # repacked buffer that /api/ps counts as size_vram, though no GPU
        # exists. A positive size_vram is accepted on the CPU cell only when
        # the server discovered the CPU and nothing else.
        cpu_only = compute == {"cpu"}
        if (args.device == "cpu" and vram != 0 and not cpu_only) or (args.device == "gpu" and vram <= 0):
            raise RuntimeError(f"Ollama placement does not match {args.device}: size_vram={vram}"
                               f", discovered compute={sorted(compute) if compute else 'unknown'}")
        if index:
            sample = {"generated_tokens": result["eval_count"], "prompt_tokens": result.get("prompt_eval_count"),
                      "request_seconds": elapsed, "load_seconds": result.get("load_duration", 0) / 1e9,
                      "text": result.get("response")}
            if isinstance(result.get("prompt_eval_cached_count"), int):
                sample["cached_prompt_tokens"] = result["prompt_eval_cached_count"]
            if result.get("eval_duration", 0) > 0:
                sample["decode_tokens_per_second"] = result["eval_count"] * 1e9 / result["eval_duration"]
            samples.append(sample)
    return {"protocol": "one warmup then measured non-streaming HTTP requests; model kept resident",
            "device_evidence": {**placement, "discovered_compute": sorted(compute) if compute else None}, "model": {"name": args.model, "digest": placement.get("digest")},
            "dtype": placement.get("details", {}).get("quantization_level"), "samples": samples}
