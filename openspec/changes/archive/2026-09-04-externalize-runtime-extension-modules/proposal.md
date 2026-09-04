## Why

`reach-architecture-freeze-1`'s design.md left one question explicitly open: "Should `externalize-runtime-extension-modules` (Change C) become a normative OpenSpec statement that Components/Formats/Providers live outside the Core repository, or stay a repository/packaging decision? Not decided here; revisit after Phase 7." The underlying architecture is no longer hypothetical — `components/qwen`, `providers/cpu`, `formats/gguf`, and `formats/safetensors` are all real, externally-hosted implementations today, pinned into this repository as git submodules, with `magnetar-runtime` verified (informally, per-module, not by one general rule) to have zero dependency on any of them. What is still missing is the normative statement itself: nothing in `openspec/specs/` currently says this externalization is required, only that it happens to be true today. Per this repository's own OpenSpec governance rule (a correct spec plus non-conformant code is an implementation issue; a new semantic decision is a new Change), formalizing "externalized by requirement, not by convention" is exactly that kind of new decision, and belongs in its own Change rather than drifted into `reach-architecture-freeze-1`, whose own proposal explicitly says not to do that.

## What Changes

- New requirement on the `project-architecture` capability: model-architecture Components, Providers (beyond one minimal built-in reference baseline), and Model Artifact Format parsers SHALL live in repositories separate from the Magnetar Core, pinned into it as git submodules, and the Core SHALL have zero compile-time dependency on any of their concrete implementation crates.
- Explicit, named exception for Reference CPU: the Core MAY retain a minimal, generic in-crate implementation for its own test suite (the "double généraliste minimal in-crate" architecture `reach-architecture-freeze-1` task group 14 already chose and implemented), distinct from the real, externally-distributable implementation in `providers/cpu` that the Core never references.
- Generalizes the CI dependency guard `reach-architecture-freeze-1`/`implement-model-format-parsers` added only for the two format crates (`.github/workflows/quality.yml`'s `submodule-integration` job) into one check covering every externalized module crate (`magnetar-component-*`, `magnetar-format-*`, `magnetar-provider-*`), closing a real gap: today nothing in CI would catch `magnetar-runtime` accidentally gaining a dependency on `providers/cpu` or `components/qwen` specifically, only on the two format crates.
- No code changes to the externalized modules themselves — they already conform; this Change makes conformance a checked requirement rather than an implicit fact.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `project-architecture`: Adds a new requirement that model-architecture Components, Providers beyond one minimal built-in reference baseline, and Model Artifact Format parsers SHALL live outside the Core repository, with the Core SHALL having zero compile-time dependency on their concrete crates.

## Impact

- Affected specs: `openspec/specs/project-architecture/spec.md` (new requirement).
- Affected code: `.github/workflows/quality.yml`'s `submodule-integration` job (the dependency-guard step generalized from two format crates to every externalized module crate).
- No changes to `components/qwen`, `providers/cpu`, `formats/gguf`, `formats/safetensors`, `components/llama`, or `providers/cuda` themselves — all already conform to the requirement being formalized.
- Compatibility: none broken. This closes an open question, generalizes an existing check, and does not change any existing behavior.
