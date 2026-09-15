<!--
Thanks for contributing to Magnetar. This template mirrors the quality
gates CONTRIBUTING.md documents -- fill in what applies, delete what
doesn't, and leave a note wherever a box can't be checked yet.
-->

## What and why

<!-- What does this change do, and why is it needed? If it fixes an issue,
reference it (e.g. "Fixes #123"). -->

## Which manifests this touches

Magnetar has one root workspace (`magnetar-runtime`) plus several
independent Cargo workspaces that a plain root-level `cargo fmt`/`check`/
`clippy`/`test` does not cover (see the root `Cargo.toml`'s own comment).
Check every one this change actually touches:

- [ ] `magnetar-runtime` (root workspace)
- [ ] `magnetar-cli`
- [ ] `inference-components`
- [ ] `integration-tests/production-loading`
- [ ] `integration-tests/multi-device-cpu-cuda`
- [ ] `integration-tests/cuda-first-native`
- [ ] `tools/coverage-ratchet`
- [ ] A submodule (`formats/*`, `components/*`, `providers/*`, `loaders/*`)

## Quality gates

Run the commands for every manifest checked above (see
[docs/quality.md](../docs/quality.md) for the full list and PowerShell
forms):

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --locked --workspace --all-targets`
- [ ] `cargo test ... --all-features` if this change touches a
      feature-gated module (`component_wasmtime`, `component_web`) --
      `--all-targets` alone does not compile these
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --all-features --no-deps`
      if this change touches imports or doc comments
- [ ] `openspec validate --all --strict` if this change touches
      `openspec/`

## Specification

- [ ] This change alters Runtime behaviour and carries a corresponding
      `openspec/` change, **or**
- [ ] This change is a refactor/fix with no behaviour change, so no spec
      change is needed

## Coverage

- [ ] This change does **not** lower `quality/coverage-baseline.json`
- [ ] This change **does** lower the baseline, and the reason is
      explained in this PR's description (a policy decision, not a
      side effect)

## Tests

- [ ] New or changed behaviour has a test that fails without the fix
- [ ] Contract-level behaviour is tested in
      `magnetar-runtime/tests/contract_tests/`; unit tests live beside
      the code they cover
