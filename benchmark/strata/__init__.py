"""Benchmark a running Strata server using its chat-completions API."""
import json
import os
import time
from common import PROMPT, http_json


def run(args, raw_dir):
    if not args.url or not args.model:
        raise ValueError("Strata requires --url and --model for an already running server")
    token = os.environ.get("STRATA_API_KEY")
    body = {"model": args.model, "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0, "max_tokens": args.tokens, "stream": False}
    raw_dir.mkdir(parents=True, exist_ok=False)
    samples = []
    for index in range(args.runs + 1):
        started = time.perf_counter()
        result = http_json(args.url.rstrip("/") + "/v1/chat/completions", body, token)
        elapsed = time.perf_counter() - started
        (raw_dir / f"run-{index}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        choices = result.get("choices") or []
        if not choices or choices[0].get("finish_reason") not in ("stop", "length"):
            raise RuntimeError("Strata returned an incomplete chat completion")
        usage = result.get("usage") or {}
        count = usage.get("completion_tokens")
        if not isinstance(count, int) or count < 1:
            raise RuntimeError("Strata did not return a positive completion_tokens count")
        if index:
            samples.append({"generated_tokens": count, "prompt_tokens": usage.get("prompt_tokens"),
                            "request_seconds": elapsed, "request_tokens_per_second": count / elapsed,
                            "finish_reason": choices[0]["finish_reason"]})
    return {"protocol": "one warmup then non-streaming chat HTTP requests; server applies its chat template",
            "device_evidence": "user-configured server placement, not independently verified",
            "model": {"name": args.model, "identity": "server configured; verify weights and revision"},
            "dtype": "server configured", "samples": samples}
