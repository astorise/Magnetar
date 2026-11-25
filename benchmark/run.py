#!/usr/bin/env python3
"""Run one engine/device cell and persist its raw evidence and measurements."""
import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import PROMPT, environment, save, summarize


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("magnetar", "candle", "llama_cpp", "llama_server", "ollama", "strata", "airllm"))
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--model-path", help="Local Hugging Face model directory or GGUF file")
    parser.add_argument("--binary", help="Path to the Candle example or llama-completion executable")
    parser.add_argument("--url", help="Base URL of the running Ollama or Strata server")
    parser.add_argument("--model", help="Model name exposed by the server")
    parser.add_argument("--server-log", help="The Ollama server's log, for the compute it discovered")
    parser.add_argument("--revision", help="Engine version or Git revision, recorded as supplied")
    parser.add_argument("--notes", help="Hardware, driver, model conversion or placement details")
    parser.add_argument("--tokens", type=positive, default=16)
    parser.add_argument("--runs", type=positive, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir / f"{stamp}-{args.engine}-{args.device}"
    raw_dir = output / "raw"
    try:
        result = importlib.import_module(args.engine).run(args, raw_dir)
        # Request throughput includes prompt processing and generation. Native
        # decode throughput remains separate because its boundaries differ.
        for sample in result["samples"]:
            if "end_to_end_latency_ms" in sample:
                sample["request_seconds"] = sample["end_to_end_latency_ms"] / 1000
            seconds = sample.get("request_seconds")
            tokens = sample.get("generated_tokens")
            if isinstance(seconds, (int, float)) and seconds > 0 and isinstance(tokens, int):
                sample["request_tokens_per_second"] = tokens / seconds
        report = {"schema": "magnetar.benchmark/v1", "engine": args.engine,
                  "requested_device": args.device, "prompt": PROMPT,
                  "max_new_tokens": args.tokens, "measured_runs": args.runs,
                  "engine_revision": args.revision, "notes": args.notes,
                  "environment": environment(), **result}
        report["summary"] = summarize(report["samples"])
        save(output / "report.json", report)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"Benchmark failed: {error}\nEvidence directory: {raw_dir}", file=sys.stderr)
        return 1
    print(json.dumps({"report": str(output / "report.json"), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
