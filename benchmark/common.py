"""Shared, dependency-free utilities for the benchmark adapters."""
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "The capital of France is"


def http_json(url, body=None, token=None, timeout=3600):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def timed_command(command, *, cwd=None, env=None):
    start = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=3600)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Command exceeded the one-hour timeout: {command!r}") from error
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command!r}\n{result.stderr[-5000:]}\n{result.stdout[-2000:]}")
    return result.stdout, result.stderr, elapsed


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    return {"os": platform.platform(), "architecture": platform.machine(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "python": platform.python_version()}


def summarize(samples):
    keys = sorted({key for sample in samples for key, value in sample.items()
                   if isinstance(value, (float, int)) and not isinstance(value, bool)})
    return {key: {"median": statistics.median(values), "min": min(values),
                  "max": max(values), "count": len(values)}
            for key in keys if (values := [sample[key] for sample in samples
                                         if isinstance(sample.get(key), (float, int))
                                         and not isinstance(sample.get(key), bool)])}


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
