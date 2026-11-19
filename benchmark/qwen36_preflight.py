"""Refuse the large experimental model before downloading when resources cannot fit it."""
import os
import shutil
import sys
from pathlib import Path

MIN_MEMORY = 30_000_000_000  # approximately a 32 GB machine
MIN_FREE_DISK = 30_000_000_000  # model, engine and compilation/download headroom


def effective_memory():
    memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            value = path.read_text(encoding="utf-8").strip()
            if value.isdigit():
                memory = min(memory, int(value))
        except OSError:
            pass
    return memory


def main(work, model):
    memory = effective_memory()
    free = shutil.disk_usage(work).free
    if memory < MIN_MEMORY:
        raise RuntimeError(f"Qwen3.6 experimental needs approximately 32 GB RAM; effective limit is {memory / 1e9:.1f} GB")
    if not Path(model).is_file() and free < MIN_FREE_DISK:
        raise RuntimeError(f"Qwen3.6 download needs at least 30 GB free disk; only {free / 1e9:.1f} GB is free")
    print(f"Qwen3.6 resource preflight: effective RAM {memory / 1e9:.1f} GB; free disk {free / 1e9:.1f} GB")


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
