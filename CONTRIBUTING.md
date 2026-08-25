# Contributing to Magnetar

## Quality gates

Every change must pass the same commands CI runs. They are listed with both
shell and PowerShell forms in [docs/quality.md](docs/quality.md). The short
version:

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-targets
```

The Rust toolchain is pinned in `rust-toolchain.toml`; rustup installs it
automatically when you run cargo in this directory. Do not install a different
version to match CI -- CI uses the same pinned toolchain.

The declared MSRV lives in `magnetar-runtime/Cargo.toml` as `rust-version` and
is verified by its own CI job. Raising it is a deliberate change, not a side
effect.

## Specifications come first

Magnetar is specification-driven. Architecture and behaviour are defined in
`openspec/` and `docs/architecture/` before they are implemented, and
`openspec validate --all --strict` runs in CI.

A change that alters runtime behaviour should carry the corresponding
specification change. A change that only refactors need not.

## Terminology

The architecture uses specific terms, and reviews will hold you to them:

- **Provider** -- trusted native runtime extension. Not "backend", not "plugin".
- **Component** -- portable WebAssembly extension. Not "plugin", not "host".
- **Capability** -- portable WIT contract a Component may import.
- **Device** -- execution target exposed by a Provider.

`README.md` has the full model.

## Tests

Contract tests live in `magnetar-runtime/tests/contract_tests/` and are the
place for anything asserting specified behaviour. Unit tests live beside the
code they cover.

A bug fix should come with a test that fails without the fix. If the test
passes against the unfixed code, it is not testing the bug.

## Coverage

The coverage ratchet in `quality/coverage-baseline.json` prevents regressions.
Lowering the baseline is a policy decision and needs to be justified in the
change that does it.

## Commits and pull requests

Explain why the change is needed, not only what it does. If a change fixes an
issue, reference it.
