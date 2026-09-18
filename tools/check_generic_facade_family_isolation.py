#!/usr/bin/env python3
"""Tachyon integration audit MAG-05 (#80): the existing `quality / model-family
isolation` guard excludes first_native_runtime.rs and qwen_model_component.rs
entirely, because those files are explicitly designated to legitimately carry
Qwen-specific knowledge (the Qwen singleton path, the Qwen test-oracle graph).
That exclusion means the guard has zero visibility into the *generic*
production facade that happens to live inside first_native_runtime.rs too --
ProductionLoadedModel, production_model_fixture,
build_first_native_graphs_from_named_component -- and inference-components'
own LoadedInferenceComponent::load, so it stayed green through MAG-01 (#82)
the whole time.

This guard scans exactly those specific bodies (not the whole file) for
model-family identifiers, source text only (comments stripped first, the same
way the existing guard already does), against an explicit, individually
justified allowlist of exceptions already tracked by #82. Anything else is a
new regression and fails the check immediately -- the point is to stop this
specific blind spot from recurring silently, not to require #82 to be fully
resolved before this guard can exist at all.

Usage: check_generic_facade_family_isolation.py [repo-root]

Exits 0 (with no output) when every scanned body's family-identifier hits are
all on the allowlist. Exits 1 with a descriptive message otherwise.
"""

import re
import sys
from pathlib import Path

FAMILY_PATTERN = re.compile(r"(?i)\b\w*(qwen|llama|gemma|mistral|mixtral|phi)\w*\b")
COMMENT_PATTERN = re.compile(r"//.*")


def extract_body(source: str, needle: str, occurrence: int, label: str) -> str:
    """Finds the `occurrence`-th (0-indexed) match of `needle` in `source` and
    returns the brace-balanced body starting at the next `{` after it -- the
    same naive-but-proven brace-counting approach this repo's own
    check_execute_first_native_graph_nodes_transport_has_no_host_tensor_typed_calls
    test uses for the identical kind of source-level static check."""
    starts = [m.start() for m in re.finditer(re.escape(needle), source)]
    if occurrence >= len(starts):
        raise SystemExit(
            f"::error::{label}: expected occurrence {occurrence} of {needle!r}, found {len(starts)}"
        )
    idx = starts[occurrence]
    body_start = source.index("{", idx)
    depth = 0
    body_end = body_start
    for offset, ch in enumerate(source[body_start:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                body_end = body_start + offset + 1
                break
    if body_end == body_start:
        raise SystemExit(f"::error::{label}: braces did not balance for {needle!r}")
    return source[body_start:body_end]


def check(label: str, body: str, allowed: set[str]) -> list[str]:
    stripped = COMMENT_PATTERN.sub("", body)
    hits = {match.group(0) for match in FAMILY_PATTERN.finditer(stripped)}
    violations = sorted(hits - allowed)
    if violations:
        return [f"::error::{label}: unallowed model-family identifier(s) {violations}"]
    return []


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    runtime_path = root / "magnetar-runtime/src/first_native_runtime.rs"
    inference_path = root / "inference-components/src/lib.rs"

    runtime_src = runtime_path.read_text(encoding="utf-8")
    inference_src = inference_path.read_text(encoding="utf-8")

    errors: list[str] = []

    # #82's own tracked, individually justified exceptions -- each already
    # investigated and confirmed to be either (a) the deliberate QwenConfig
    # bridge every ModelArchitectureConfig-driven caller still needs until
    # E2eFixture itself stops being QwenConfig-typed, or (b) the intentional,
    # documented hardcoded-Qwen-singleton fallback for a None component_digest.
    errors += check(
        "production_model_fixture",
        extract_body(runtime_src, "fn production_model_fixture(", 0, "production_model_fixture"),
        allowed={
            "qwen_config_from_architecture_config",
            "qwen_component_descriptor",
            "qwen_validate_model_artifact",
        },
    )
    errors += check(
        "impl ProductionLoadedModel",
        extract_body(runtime_src, "impl ProductionLoadedModel {", 0, "impl ProductionLoadedModel"),
        allowed={
            "architecture_config_from_qwen_config",
            "build_first_native_graphs_from_real_qwen_component",
        },
    )

    # build_first_native_graphs_from_named_component has two cfg-gated
    # definitions (the real one, and the fail-closed stub for builds without
    # the wasmtime-component-engine feature) -- both must stay clean.
    for occurrence in (0, 1):
        errors += check(
            f"build_first_native_graphs_from_named_component (definition {occurrence})",
            extract_body(
                runtime_src,
                "pub fn build_first_native_graphs_from_named_component(",
                occurrence,
                "build_first_native_graphs_from_named_component",
            ),
            allowed=set(),
        )

    errors += check(
        "LoadedInferenceComponent::load",
        extract_body(inference_src, "pub fn load(", 0, "LoadedInferenceComponent::load"),
        allowed=set(),
    )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(
            "\nThe generic production facade (ProductionLoadedModel, production_model_fixture, "
            "build_first_native_graphs_from_named_component, LoadedInferenceComponent::load) "
            "must not depend on a model-family-specific identifier unless it is an explicit, "
            "individually justified exception in tools/check_generic_facade_family_isolation.py's "
            "own allowlist -- see #82.",
            file=sys.stderr,
        )
        return 1

    print("generic facade family isolation: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
