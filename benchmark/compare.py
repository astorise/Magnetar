#!/usr/bin/env python3
"""Display selected metrics from benchmark report files without ranking unlike protocols."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    print("Metrics are engine-specific; check model, dtype, device and protocol before comparing.\n")
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema") != "magnetar.benchmark/v1":
            parser.error(f"Not a benchmark report: {path}")
        print(f"{report['engine']} / {report['requested_device']}  ({path})")
        print(f"  protocol: {report['protocol']}")
        print(f"  model: {json.dumps(report['model'], ensure_ascii=False)}")
        for key, value in report["summary"].items():
            print(f"  {key}: median={value['median']:.4g}, n={value['count']}")
        print()


if __name__ == "__main__":
    main()
