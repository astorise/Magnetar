# Implementation Evidence

Recorded: 2026-08-30T17:54:14+02:00

Baseline source commit before this working-tree change:

```text
fcca699c5f6b2ce8dad1e0ca800ad73fb90c10ed
```

## Implemented Path

The normal CLI generation path now calls
`magnetar_runtime::run_first_native_fixture_generation` from
`magnetar-cli/src/pipeline.rs`.

The prior CLI placeholder logits executor was removed from the normal path.
Runtime generation requires execution evidence emitted by the first-native
fixture path:

- execution graph validated
- kernel selected
- kernel dispatched
- Reference CPU Provider executed
- Runtime-owned tensor logits produced
- token generated

## Local Gates

The following commands were run successfully:

```text
cargo fmt --all
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
openspec validate define-first-native-model-implementation-and-conformance-cut --strict
wasm-tools component wit <each *.wit>
wasm-tools parse/validate <each magnetar-runtime fixture component *.wat>
cargo deny --all-features check
```

`cargo deny` completed successfully and reported warnings for existing duplicate
dependency versions and one unmatched license allowance.

## Coverage Evidence

`cargo test --workspace` passed:

- `magnetar-cli`: 60 unit tests
- `magnetar-runtime`: 981 unit tests
- `magnetar-runtime` contract tests: 176 tests

The runtime E2E conformance suite includes checks for:

- deterministic Qwen-like fixture identity and weight digest
- Model Artifact and Model Loading traversal
- Qwen Component boundary and authority constraints
- Execution Graph production and execution
- Kernel Registry selection
- Reference CPU Provider execution
- Tensor Resource and Memory Manager ownership
- KV/prefix cache lifecycle and redaction
- greedy and seeded sampling
- cancellation, timeout, missing kernel, invalid graph, invalid tensor, and
  trust failures
- no direct Provider/Kernel/model-loading/memory-manager bypass evidence

## Remaining External Action

This working tree still needs a commit if the project wants the final baseline
reference to be a new immutable Git commit rather than the pre-change source
commit recorded above.
