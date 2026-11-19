"""Summarize a single-host campaign without hiding incomplete engines."""
import json
import sys
from pathlib import Path


def render(root):
    profile_path = root / "model-profile.txt"
    profile = profile_path.read_text(encoding="utf-8").strip() if profile_path.exists() else "qwen2.5-0.5b"
    statuses = dict(line.split("\t", 1) for line in (root / "status.tsv").read_text(encoding="utf-8").splitlines())
    reports = {}
    for path in root.glob("*/report.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema") == "magnetar.benchmark/v1":
            reports[report["engine"]] = report
    lines = ["# Single-machine inference campaign", "", f"Model profile: `{profile}`", "",
             "| Engine | Status | Prompt tokens | Generated tokens | Request latency median (s) | Request throughput median (tokens/s) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    def median(summary, key):
        value = summary.get(key, {}).get("median")
        return f"{value:.3f}" if value is not None else "—"
    for engine, status in statuses.items():
        summary = reports.get(engine, {}).get("summary", {})
        lines.append(f"| {engine} | {status} | {median(summary, 'prompt_tokens')} | {median(summary, 'generated_tokens')} | "
                     f"{median(summary, 'request_seconds')} | {median(summary, 'request_tokens_per_second')} |")
    prompt_counts = {engine: r.get("summary", {}).get("prompt_tokens", {}).get("median") for engine, r in reports.items()}
    generated_counts = {engine: r.get("summary", {}).get("generated_tokens", {}).get("median") for engine, r in reports.items()}
    if len({value for value in prompt_counts.values() if value is not None}) > 1 or len({value for value in generated_counts.values() if value is not None}) > 1:
        lines.extend(["", "**Token counts differ across engines; the requests are not directly equivalent.**"])
    if any(value is None for value in prompt_counts.values()):
        lines.extend(["", "**Prompt token counts are unavailable for some engines; equivalence is not verified.**"])
    cached = reports.get("ollama", {}).get("summary", {}).get("cached_prompt_tokens", {}).get("max")
    if cached:
        lines.extend(["", f"**Ollama reused up to {cached} prompt tokens from cache; prefill work differs.**"])
    cached = reports.get("strata_native", {}).get("summary", {}).get("cached_prompt_tokens", {}).get("max")
    if cached:
        lines.extend(["", f"**Strata reused up to {cached} prompt tokens from cache; prefill work differs.**"])
    sequences = {engine: {tuple(s["generated_token_ids"]) for s in r.get("samples", [])
                          if isinstance(s.get("generated_token_ids"), list)} for engine, r in reports.items()}
    sequences = {engine: values for engine, values in sequences.items() if values}
    if len(sequences) > 1 and len(set.union(*sequences.values())) > 1:
        lines.extend(["", "**Generated token sequences differ across measured runs or engines; inspect the raw outputs.**"])
    texts = {s["text"] for r in reports.values() for s in r.get("samples", []) if isinstance(s.get("text"), str)}
    if len(texts) > 1:
        lines.extend(["", "**Generated text differs across measured runs or engines; inspect the raw outputs.**"])
    lines += ["", "These are per-request measurements after warmup on the same job host; HTTP and in-process boundaries differ."]
    if profile == "qwen3.6-35b-experimental":
        lines += ["Experimental profile: llama.cpp and Strata use the same pinned GGUF and prompt token IDs; "
                  "GPU runs use different hybrid CPU/GPU placements.",
                  "Other engines are not integrated for this architecture. This is not yet an all-engine comparison."]
    else:
        lines += ["The HF checkpoint and FP16 GGUF derive from the same model, but weight and compute precision differ; inspect each report.",
                  "Candle reloads the model for each process; request latency is estimated from its rounded prefill-plus-generation rate.",
                  "Ollama may reuse prompt cache. Check actual token counts, outputs, GPU placement, versions, logs and raw reports before ranking.",
                  "Strata is not benchmarked against Qwen2.5 because that model is not documented as supported."]
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(Path(sys.argv[1])))
