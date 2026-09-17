#!/usr/bin/env python3
"""Enforces #54's separate request: a change to quality/coverage-baseline.json's
recorded line_coverage_percent/function_coverage_percent must come with a new,
consistent entry in the file's rebaseline_history array, rather than only a
prose note appended to notes.

Usage: check_coverage_baseline_history.py <new-baseline.json> <old-baseline.json>

Exits 0 (with no output) when the recorded percentages did not change, or when
they changed and a valid new history entry justifies it. Exits 1 with a
descriptive message otherwise.
"""

import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <new-baseline.json> <old-baseline.json>", file=sys.stderr)
        return 2

    new_path, old_path = sys.argv[1], sys.argv[2]
    with open(new_path, encoding="utf-8") as f:
        new = json.load(f)
    with open(old_path, encoding="utf-8") as f:
        old = json.load(f)

    changed = old.get("line_coverage_percent") != new.get("line_coverage_percent") or old.get(
        "function_coverage_percent"
    ) != new.get("function_coverage_percent")
    if not changed:
        print("recorded line_coverage_percent/function_coverage_percent unchanged; no history entry required")
        return 0

    old_history = old.get("rebaseline_history") or []
    new_history = new.get("rebaseline_history") or []
    if len(new_history) <= len(old_history):
        print(
            "::error file=quality/coverage-baseline.json::"
            "line_coverage_percent/function_coverage_percent changed but "
            "rebaseline_history gained no new entry (see #54)"
        )
        return 1

    latest = new_history[-1]
    recorded = latest.get("recorded", {})
    if recorded.get("line_coverage_percent") != new.get("line_coverage_percent") or recorded.get(
        "function_coverage_percent"
    ) != new.get("function_coverage_percent"):
        print(
            "::error file=quality/coverage-baseline.json::"
            "rebaseline_history's latest entry's 'recorded' figures do not "
            "match the new top-level line_coverage_percent/function_coverage_percent"
        )
        return 1

    if not latest.get("reason"):
        print(
            "::error file=quality/coverage-baseline.json::"
            "rebaseline_history's latest entry is missing a non-empty 'reason'"
        )
        return 1

    print("rebaseline_history entry present and consistent with the recorded change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
