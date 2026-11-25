"""Magnetar's existing resident-model qualification campaign."""
import json
import os
from pathlib import Path

from common import ROOT, timed_command


def run(args, raw_dir):
    if args.tokens < 2:
        raise ValueError("Magnetar qualification requires --tokens >= 2 to measure decode latency")
    if not args.model_path or not Path(args.model_path).is_dir():
        raise ValueError("Magnetar requires --model-path pointing to the pinned Hugging Face bundle")
    raw_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"MAGNETAR_QWEN_REAL_CHECKPOINT_DIR": str(Path(args.model_path).resolve()),
                "MAGNETAR_QUALIFICATION_REPORT_DIR": str(raw_dir.resolve()),
                "MAGNETAR_QUALIFICATION_MAX_NEW_TOKENS": str(args.tokens),
                "MAGNETAR_QUALIFICATION_MEASURED_RUNS": str(args.runs),
                "MAGNETAR_QUALIFICATION_PROVIDERS": "cpu-native" if args.device == "cpu" else "cuda"})
    if args.device == "gpu":
        env["MAGNETAR_QUALIFICATION_REQUIRE_CUDA"] = "1"
    else:
        env.pop("MAGNETAR_QUALIFICATION_REQUIRE_CUDA", None)
    env.pop("MAGNETAR_QUALIFICATION_CONTEXT_TOKENS", None)
    env.pop("MAGNETAR_REAL_CHECKPOINT_PROFILE", None)
    env.pop("MAGNETAR_QUALIFICATION_SUMMARY_PATH", None)
    command = ["cargo", "test", "--release", "--manifest-path",
               str(ROOT / "integration-tests/production-loading/Cargo.toml"),
               "real_checkpoint_qualification_campaign", "--", "--include-ignored", "--nocapture"]
    stdout, stderr, elapsed = timed_command(command, cwd=ROOT, env=env)
    (raw_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (raw_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    reports = list(raw_dir.glob("*.json"))
    if len(reports) != 1:
        raise RuntimeError(f"Expected one Magnetar campaign report, got {len(reports)}")
    campaign = json.loads(reports[0].read_text(encoding="utf-8"))
    samples = []
    for sample in campaign["samples"]:
        if sample["phase"] == "measured":
            metrics = dict(sample["metrics"])
            if "generated_token_ids" in sample:
                metrics["generated_token_ids"] = sample["generated_token_ids"]
            samples.append(metrics)
    if len(samples) != args.runs:
        raise RuntimeError("Magnetar report did not contain the requested measured runs")
    return {"protocol": "resident model; cold request, one warmup, measured streaming requests",
            "device_evidence": campaign["manifest"]["provider"],
            "model": campaign["manifest"]["checkpoint"], "dtype": campaign["manifest"]["dtype"],
            "samples": samples, "campaign_report": str(reports[0]), "command_seconds": elapsed}
